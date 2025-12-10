import pandas as pd
import numpy as np
from pprint import pprint

# These are the square endpoints from USGS_dbscan.py
SQUARE_MIN_LAT = 7.4210
SQUARE_MAX_LAT = 9.8038
SQUARE_MIN_LON = 125.7307
SQUARE_MAX_LON = 128.1135

# This is the output file from USGS_dbscan.py with 202312Dataset.csv as input
INPUT_FILE = "202312Spacial.csv"

L = 500

# Load .csv file
df = pd.read_csv(INPUT_FILE)
points = df[['latitude', 'longitude', 'mag']].to_numpy()
lats = points[:, 0]
lons = points[:, 1]
mags = points[:, 2]

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
