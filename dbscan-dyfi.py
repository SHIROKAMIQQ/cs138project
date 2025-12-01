import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# This acts as the threshold for considering how close cells must be to each other
#  to be considered part of the same neighborhood/cluster.
EPS_KM = 30
EARTH_RADIUS_KM = 6371.0
EPS_RAD = EPS_KM / EARTH_RADIUS_KM

# Load .csv file
# This assumes dyfi for now. 
csv_file = "dyfi.csv"  # replace with your file path
df = pd.read_csv(csv_file)
coords = df[['Latitude', 'Longitude']].to_numpy()

# DBSCAN clustering using Haversine distance
# https://stackoverflow.com/questions/24762435/clustering-geo-location-coordinates-lat-long-pairs-using-kmeans-algorithm-with
# https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/ 
coords_rad = np.radians(coords)  # lat/lon in radians
db = DBSCAN(eps=EPS_RAD, min_samples=3, metric='haversine')
# print("DB:", db)
labels = db.fit_predict(coords_rad)
# print("labels:", labels)
df['Cluster'] = labels

# Find largest cluster/neighborhood
unique_labels = [l for l in set(labels) if l != -1]  # exclude noise
if not unique_labels:
    raise ValueError("No clusters found. Try increasing eps or decreasing min_samples.")
cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}
largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
print(f"Largest cluster: {largest_cluster_label} with {cluster_sizes[largest_cluster_label]} points")

# Extract rows of largest cluster to dyfi_largest_cluster.csv
largest_cluster_rows = df[df['Cluster'] == largest_cluster_label]
largest_cluster_rows.to_csv("dyfi_largest_cluster.csv", index=False)
print("Saved largest cluster to dyfi_largest_cluster.csv")

# Compute square in terms of lat/lng
lat_min = largest_cluster_rows['Latitude'].min()
lat_max = largest_cluster_rows['Latitude'].max()
lon_min = largest_cluster_rows['Longitude'].min()
lon_max = largest_cluster_rows['Longitude'].max()
lat_range = lat_max - lat_min
lon_range = lon_max - lon_min
max_range = max(lat_range, lon_range)

# Build square bounding box (equal sides)
lat_mid = (lat_max + lat_min)/2
lon_mid = (lon_max + lon_min)/2
half_size = max_range/2

square_lat_min = lat_mid - half_size
square_lat_max = lat_mid + half_size
square_lon_min = lon_mid - half_size
square_lon_max = lon_mid + half_size

print("Square bounding box (lat/lng degrees):")
print(f"Latitude:  {square_lat_min:.4f} to {square_lat_max:.4f}")
print(f"Longitude: {square_lon_min:.4f} to {square_lon_max:.4f}")

# Plot
plt.figure(figsize=(8,6))
colors = plt.cm.get_cmap('tab20', len(unique_labels))

for k in unique_labels:
    mask = (labels == k)
    xy = coords[mask]
    plt.scatter(xy[:,1], xy[:,0], label=f'Cluster {k}')

# noise points
mask_noise = (labels == -1)
plt.scatter(coords[mask_noise,1], coords[mask_noise,0], c='k', marker='x', label='Noise')

# overlay square of largest cluster
square_lon = [square_lon_min, square_lon_max, square_lon_max, square_lon_min, square_lon_min]
square_lat = [square_lat_min, square_lat_min, square_lat_max, square_lat_max, square_lat_min]
plt.plot(square_lon, square_lat, 'r-', linewidth=2, label='Bounding Square (Largest Cluster)')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering of DYFI Responses with Square Bounding Box')
plt.legend()
plt.show()
