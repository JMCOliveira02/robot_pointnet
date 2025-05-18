import numpy as np
from sklearn.neighbors import KDTree
import time

# Step 1: Load the points
def create_kd_tree():
    wall_label = 0
    ceiling_label = 1
    floor_label = 2
    wall_points = np.load('/home/joao/dev/datasets/iilab/wall_points.npy')        # e.g., label 0
    ceiling_points = np.load('/home/joao/dev/datasets/iilab/ceiling_points.npy')   # e.g., label 1
    floor_points = np.load('/home/joao/dev/datasets/iilab/floor_points.npy')       # e.g., label 2
    wall_labels = np.full(wall_points.shape[0], wall_label)      # All walls labeled 0
    ceiling_labels = np.full(ceiling_points.shape[0], ceiling_label) # All ceilings labeled 1
    floor_labels = np.full(floor_points.shape[0], floor_label)     # All floors labeled 2

    all_points = np.vstack([wall_points, ceiling_points, floor_points])
    all_labels = np.concatenate([wall_labels, ceiling_labels, floor_labels])

    kd_tree = KDTree(all_points)
    return kd_tree, all_labels

# Now you can use the KD-Tree for queries
kd_tree, all_labels = create_kd_tree()
# Example query
x = 0.0
y = 0.0
z = 2.0
query_point = np.array([[x, y, z]])  # Adjust the coordinates as needed
time_start = time.time()
time_end = time.time()
print(f"Query time: {time_end - time_start:.6f} seconds")

dist, ind = kd_tree.query(query_point, k=1)
nearest_label = all_labels[ind[0][0]]

print(f"Nearest point to {x, y, z} is labeled as {nearest_label}")
