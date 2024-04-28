from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Type
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from torch.nn import Parameter
import re
from f3rm.feature_field import FeatureField, FeatureFieldHeadNames
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer
from f3rm.features.clip import tokenize
import json
from f3rm.prompt import get_data_json, get_response,select_cluster_prompt,semantic_parsing
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    # Weighing for the feature loss
    feat_loss_weight: float = 1e-3
    # Feature Field Positional Encoding
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    # Feature Field Hash Grid
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    # Feature Field MLP Head
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2


@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def clip(self):
        from f3rm.features.clip import load
        from f3rm.features.clip_extract import CLIPArgs

        CONSOLE.print(f"Loading CLIP {CLIPArgs.model_name} for viewer")
        model, _ = load(CLIPArgs.model_name, device=self.device)
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        from f3rm.features.clip import tokenize

        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        print("query shape: ",embed.shape)
        if is_positive:
            self.positives = texts
            print("texts: ",texts)
            self.positives_target_reference_dict={key:{} for key in texts}
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed /= embed.norm(dim=-1, keepdim=True)
            self.pos_embed = embed
            print("Weighted query shape: ",embed.shape)
        else:
            self.negatives = texts
            # We don't average the negatives as we compute pair-wise softmax
            embed /= embed.norm(dim=-1, keepdim=True)
            self.neg_embed = embed

    @property
    def has_positives(self) -> bool:
        return self.positives and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return self.negatives and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils()


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    renderer_feature: FeatureRenderer

    def populate_modules(self):
        super().populate_modules()

        # Create feature field
        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError(f"Feature dimensionality must be positive, not {feature_dim}")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
        )
        self.renderer_feature = FeatureRenderer()
        self.setup_gui()

    def setup_gui(self):
        viewer_utils.device = self.kwargs["device"]
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        if self.kwargs["metadata"]["feature_type"] != "CLIP":
            return
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=viewer_utils.softmax_temp,
            cb_hook=lambda elem: viewer_utils.update_softmax_temp(elem.value),
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Feature outputs
        ff_outputs = self.feature_field(ray_samples)
        features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }
        torch.save(outputs,"./output.pth")
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Compute feature error
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Compute feature loss
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)
        return loss_dict

    @torch.no_grad()
    def process_sims(self,viewer_utils:ViewerUtils,outputs,clip_features:torch.Tensor)->torch.Tensor:
        texts=viewer_utils.positives
        print("Process Sims texts: ",texts)
        weighted_sims=torch.zeros(outputs['rgb'].shape[0],outputs['rgb'].shape[1],1).cuda()
        for query in texts:
            if viewer_utils.positives_target_reference_dict[query]=={}:
                prompt=semantic_parsing.format(query)
                payload = get_data_json(image_tensor=outputs['rgb'], image_path=None,prompt=prompt, max_tokens=200)
                print(prompt)
                response = json.loads(get_response(payload)['choices'][0]['message']['content'])
                print(response)
                target_object=response['target_object']
                reference_object=response['reference_object']
                viewer_utils.positives_target_reference_dict[query]={"target":target_object,"reference":reference_object}
            target_object,reference_object=viewer_utils.positives_target_reference_dict[query]["target"],viewer_utils.positives_target_reference_dict[query]["reference"]
            target_tokens = tokenize(target_object).to(viewer_utils.device)
            target_embed = viewer_utils.clip.encode_text(target_tokens).float()
            if reference_object is not None and reference_object!='None':
                reference_tokens=tokenize(reference_object).to(viewer_utils.device)
                reference_embed=viewer_utils.clip.encode_text(reference_tokens).float()
                reference_sims=clip_features@reference_embed.T
                reference_cluster_centers=self.reference_cluster(reference_sims,shape=outputs['rgb'].shape[0:2])
            else:
                reference_cluster_centers=None
            target_sims=clip_features @ target_embed.T
            # torch.save(reference_sims,"./f3rm/reference_sim.pth")
            # torch.save(target_sims,"./f3rm/target_sim.pth")
            image_tensor=self.render_image_view(outputs['rgb'])
            weighted_sims=weighted_sims+self.target_cluster(target_sims,reference_cluster_centers,target_object,reference_object,query,image=image_tensor,shape=outputs['rgb'].shape[0:2])
        return (weighted_sims/len(texts))

    def render_image_view(self,rgb_info):
        image_tensor=rgb_info.clone().cpu()
        image_tensor_scaled = image_tensor.float() * 255
        image_array_rgb = image_tensor_scaled .numpy().astype(np.uint8)
        image_array_symmetric = np.flip(image_array_rgb, axis=0)
        fig, ax = plt.subplots()
        ax.imshow(image_array_symmetric)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, image_array_symmetric.shape[1], 50))
        ax.set_yticks(np.arange(0, image_array_symmetric.shape[0], 50))
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
        plt.tight_layout()
        fig.canvas.draw()
        image_plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_plot_array = image_plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_plot_tensor = torch.tensor(image_plot_array.copy()) 
        plt.close()
        return image_plot_tensor

    def reference_cluster(self,reference_sims,shape)->torch.Tensor:
        # Compute threshold
        reference_sims=reference_sims.cpu()
        threshold = reference_sims.max() * 0.9
        # Find indices of similarities above the threshold
        indices_in_interest = torch.where(reference_sims > threshold)
        coordinates_in_interest = np.column_stack((indices_in_interest[0].numpy(), indices_in_interest[1].numpy()))
        # Scale the coordinates
        scaler = StandardScaler()
        scaled_coordinates = scaler.fit_transform(coordinates_in_interest)
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)  
        cluster_labels = dbscan.fit_predict(scaled_coordinates)
        # Calculate centers of coordinates for each cluster label
        cluster_centers = []
        for label in np.unique(cluster_labels):
            if label == -1:
                continue  # Skip noise points
            cluster_center = np.mean(scaled_coordinates[cluster_labels == label], axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = scaler.inverse_transform(np.array(cluster_centers))
        GPT_cluster_centers=[[round(y),shape[1]-round(x)] for [x,y] in cluster_centers]
        print("GPT_reference_cluster: ",GPT_cluster_centers)
        return GPT_cluster_centers

    def target_cluster(self,target_sims,reference_cluster_centers,target_object,reference_object,query,image,shape)->torch.Tensor:
        # Compute threshold
        target_sims=target_sims.cpu()
        threshold = target_sims.max() * 0.9
        # Find indices of similarities above the threshold
        indices_in_interest = torch.where(target_sims > threshold)
        coordinates_in_interest = np.column_stack((indices_in_interest[0].numpy(), indices_in_interest[1].numpy()))
        # Scale the coordinates
        scaler = StandardScaler()
        scaled_coordinates = scaler.fit_transform(coordinates_in_interest)
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)  
        cluster_labels = dbscan.fit_predict(scaled_coordinates)
        # Calculate centers of coordinates for each cluster label
        cluster_centers = []
        for label in np.unique(cluster_labels):
            if label == -1:
                continue  # Skip noise points
            cluster_center = np.mean(scaled_coordinates[cluster_labels == label], axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = scaler.inverse_transform(np.array(cluster_centers))

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        positions = np.stack([y, x], axis=-1)
        # Calculate Euclidean distances to each cluster center
        distances = np.sum(np.abs(positions[:, :, None, :] - cluster_centers[None, None, :, :])**2, axis=3)
        d_list=[]
        for i in range(distances.shape[2]):
            d_list.append(distances[...,i])
        if len(d_list)>1:
            for i in range(distances.shape[2]):
                d_list.append(distances[...,i])
            GPT_cluster_centers=[[round(y),shape[0]-round(x)] for [x,y] in cluster_centers]
            GPT_target_cluster_center_prompt=" ".join([f"{i}. {GPT_cluster_centers[i]}" for i in range(len(GPT_cluster_centers))])
            if reference_cluster_centers is not None:
                GPT_reference_cluster_center_prompt=" ".join([f"{i}. {reference_cluster_centers[i]}" for i in range(len(reference_cluster_centers))])
                [x,y]=reference_cluster_centers[0]
                gap=" ".join([f"{i}. {[abs(a-x),abs(b-y)]}" for (i,[a,b]) in enumerate(GPT_cluster_centers)])
            else:
                GPT_reference_cluster_center_prompt=None
                gap=None
            prompt=select_cluster_prompt.format(query,target_object,reference_object,GPT_target_cluster_center_prompt,GPT_reference_cluster_center_prompt,gap)
            print(prompt)
            payload = get_data_json(image_tensor=image, image_path=None,prompt=prompt, max_tokens=200)
            response = get_response(payload)['choices'][0]['message']['content']
            print(response)
            try:
                selected_id=int(json.loads(response)['target_object_cluster'])
            except:
                match = re.search(r'"target_object_cluster":"(\d+)"', response)
                selected_id = int(match.group(1))
            weights = d_list[selected_id] / sum(d_list)
            weights = np.nan_to_num(weights)
            weights = weights[:, :, None]
            weights=1-(weights-weights.min())/(weights.max()-weights.min())
        else:
            weights=1
        weighted_sims = target_sims * weights
        return weighted_sims.cuda()

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not viewer_utils.has_negatives:
            
            # print("texts: ",viewer_utils.positives)
            print("Calling Again")
            torch.save(outputs["rgb"],"rgb_info.pth")
            
            # sims = clip_features @ viewer_utils.pos_embed.T
            # print("get_outputs_for_camera_ray_bundle Clip Features Shape: ",clip_features.shape)
            # print("get_outputs_for_camera_ray_bundle Query Embedding Shape: ",viewer_utils.pos_embed.shape)
            # Show the mean similarity if there are multiple positives
            # if sims.shape[-1] > 1:
            #     sims = sims.mean(dim=-1, keepdim=True)
            # # print("get_outputs_for_camera_ray_bundle Similarity shape: ",sims.shape)
            H,W,_=outputs['rgb'].shape
            if H > 100:
                sims=self.process_sims(viewer_utils,outputs,clip_features)
                outputs["similarity"] = sims
            else:
                outputs["similarity"]=clip_features @ viewer_utils.pos_embed.T
            return outputs

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / viewer_utils.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        outputs["similarity"] = sims
        return outputs
