import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load similarity matrix
sims = torch.load("./reference_sim.pth").cpu()
print(sims.shape)

# Compute threshold
threshold = sims.max() * 0.9

# Find indices of similarities above the threshold
indices_in_interest = torch.where(sims > threshold)
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
GPT_cluster_centers=[[292-round(x),round(y)] for [x,y] in cluster_centers]
print(GPT_cluster_centers)
x, y = np.meshgrid(np.arange(512), np.arange(292))
positions = np.stack([y, x], axis=-1)

# Calculate Euclidean distances to each cluster center
distances = np.sum(np.abs(positions[:, :, None, :] - cluster_centers[None, None, :, :])**2, axis=3)
d_list=[]
if len(d_list)>1:
    for i in range(distances.shape[2]):
        d_list.append(distances[...,i])
    selected_id=0
    weights = d_list[selected_id] / sum(d_list)
    weights = np.nan_to_num(weights)
    weights = weights[:, :, None]
    weights=1-(weights-weights.min())/(weights.max()-weights.min())
else:
    weights=1
weighted_sims = sims * weights
print(weighted_sims.shape)
plt.figure(figsize=(10, 5))
plt.imshow(weighted_sims, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Similarity Heatmap')
plt.savefig('./Similarity_Heatmap2.png') 