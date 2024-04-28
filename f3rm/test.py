import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

sims=torch.load("./reference_sim.pth").cpu()
print(sims.shape)
print(sims.max())
threshold = sims.max() * 0.9
indices_in_interest = torch.where(sims > threshold)
coordinates_in_interest = list(zip(indices_in_interest[0].numpy(), indices_in_interest[1].numpy()))
# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0,n_init='auto')
kmeans.fit(coordinates_in_interest)
cluster_labels = kmeans.labels_
cluster_centers=kmeans.cluster_centers_

GPT_cluster_centers=[[292-round(x),round(y)] for [x,y] in cluster_centers]
print(GPT_cluster_centers)

x, y = np.meshgrid(np.arange(512), np.arange(292))
positions = np.stack([y, x], axis=-1)

# Calculate Manhattan distances to each cluster center
distances = np.sum(np.abs(positions[:, :, None, :] - cluster_centers[None, None, :, :])**2, axis=3)
d_list=[]
for i in range(distances.shape[2]):
    d_list.append(distances[...,i])
selected_id=2
weights = d_list[selected_id] / sum(d_list)
weights = np.nan_to_num(weights)
weights = weights[:, :, None]
print(weights.max(),weights.min())
weights=1-(weights-weights.min())/(weights.max()-weights.min())
weighted_sims = sims * weights
plt.figure(figsize=(10, 5))
# plt.imshow(weighted_sims, cmap='coolwarm', interpolation='nearest')
plt.imshow(sims, cmap='coolwarm', interpolation='nearest')
# plt.imshow(sims, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Similarity Heatmap')
plt.savefig('./Similarity_Heatmap2.png') 