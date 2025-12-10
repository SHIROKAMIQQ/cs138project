import pandas as pd
import numpy as np
from pprint import pprint

# This is the output file from USGS_dbscan.py with 202312Dataset.csv as input
INPUT_FILE = "202312Spacial.csv"
L = 50

# Load .csv file
df = pd.read_csv(INPUT_FILE)
points = df[['latitude', 'longitude', 'mag']].to_numpy()
lats = points[:, 0]
lons = points[:, 1]
mags = points[:, 2]

# Compute square in terms of lat/lon
lat_min = lats.min()
lat_max = lats.max()
lon_min = lons.min()
lon_max = lons.max()
lat_range = lat_max - lat_min
lon_range = lon_max - lon_min
max_range = max(lat_range, lon_range)
# Build square bounding box (equal sides)
lat_mid = (lat_max + lat_min)/2
lon_mid = (lon_max + lon_min)/2
half_size = max_range/2
SQUARE_MIN_LAT = lat_mid - half_size
SQUARE_MAX_LAT = lat_mid + half_size
SQUARE_MIN_LON = lon_mid - half_size
SQUARE_MAX_LON = lon_mid + half_size

def discretize_axis(values, axis_min, axis_max):
  partial = (values - axis_min) / (axis_max - axis_min)
  idx = (partial * L).astype(int)
  return np.clip(idx, 0, L-1) # np.clip limits idx to (0, L-1). Treat edges as part of bordering cell.
mapped_rows = discretize_axis(lats, SQUARE_MIN_LAT, SQUARE_MAX_LAT)
mapped_cols = discretize_axis(lons, SQUARE_MIN_LON, SQUARE_MAX_LON)

# Initialize the L x L matrix
# This L x L matrix is a matrix of magnitudes. 
# If multiple earthquakes discretize into the same cell, get the mean.
mag_sum = np.zeros((L, L))
mag_count = np.zeros((L, L))
mag_matrix = np.zeros((L, L))

# Map all earthquakes to their cells
for r, c, m in zip(mapped_rows, mapped_cols, mags):
  mag_sum[r, c] += m
  mag_count[r, c] += 1

# Get means of magnitudes per cell
mask = mag_count > 0
mag_matrix[mask] = mag_sum[mask] / mag_count[mask]

pprint(mag_matrix)
