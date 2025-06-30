import numpy as np
import torch
import struct
import time

colors_segmentation = np.array([ 
        [255, 0, 0],       # 0 - ceiling      - red
        [0, 255, 0],       # 1 - floor        - green
        [0, 0, 255],       # 2 - wall         - blue
        [0, 0, 0],#[255, 255, 0],     # 3 - beam         - yellow
        [0, 0, 0],#[255, 0, 255],     # 4 - column       - magenta
        [0, 0, 0],#[0, 255, 255],     # 5 - window       - cyan
        [0, 0, 0],#[192, 192, 192],   # 6 - door         - light gray
        [0, 0, 0],#[128, 128, 128],   # 7 - table        - gray
        [0, 0, 0],#[128, 0, 0],       # 8 - chair        - maroon
        [0, 0, 0],#[128, 128, 0],     # 9 - sofa         - olive
        [0, 0, 0],#[0, 128, 0],       # 10 - bookcase    - dark green
        [0, 0, 0],#[128, 0, 128],     # 11 - board       - purple
        [0, 0, 0]          # 12 - clutter     - black
    ])

def unpack_rgb(rgb_float):
    """Unpack RGB values from a float"""
    # Convert float back to uint32
    rgb_uint = struct.unpack('I', struct.pack('f', rgb_float))[0]
    
    # Extract BGR components (note: your packing uses BGR order)
    r = rgb_uint & 0xFF
    g = (rgb_uint >> 8) & 0xFF
    b = (rgb_uint >> 16) & 0xFF
    
    return [r, g, b] # Return as RGB

def split_pointcloud_into_blocks_with_color_gpt(pc, block_size=1.0, stride=0.5, block_points=4096, padding=0.001):
    """
    Splits a colored point cloud (XYZ + RGB) into blocks.
    Returns:
        - block_data: shape (num_blocks, block_points, 9)
        - point_indices: shape (num_blocks, block_points)
    """
    coord = pc[:, :3]
    color = pc[:, 3:6] / 255.0  # Normalize RGB

    coord_min = np.amin(coord, axis=0)[:3]
    coord_max = np.amax(coord, axis=0)[:3]
    grid_x = int(np.ceil((coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil((coord_max[1] - coord_min[1] - block_size) / stride) + 1)

    all_data = []
    all_indices = []

    for idx_y in range(grid_y):
        for idx_x in range(grid_x):
            s_x = coord_min[0] + idx_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + idx_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size

            block_idxs = np.where(
                (coord[:, 0] >= s_x - padding) & (coord[:, 0] <= e_x + padding) &
                (coord[:, 1] >= s_y - padding) & (coord[:, 1] <= e_y + padding)
            )[0]

            if block_idxs.size == 0:
                continue

            # Sample or pad
            point_size = int(np.ceil(block_idxs.size / block_points) * block_points)
            replace = point_size > block_idxs.size
            sample_idxs = np.random.choice(block_idxs, point_size, replace=replace)
            np.random.shuffle(sample_idxs)
            sampled_coord = coord[sample_idxs]
            sampled_color = color[sample_idxs]

            # Normalize
            coord_centered = sampled_coord - np.mean(sampled_coord, axis=0)
            coord_normalized = coord_centered / (np.max(coord_centered) + 1e-6)

            block_feat = np.concatenate((coord_centered, sampled_color, coord_normalized), axis=1)
            block_feat = block_feat.reshape(-1, block_points, 9)

            all_data.append(block_feat)
            all_indices.append(sample_idxs.reshape(-1, block_points))

    return np.vstack(all_data), np.vstack(all_indices)

def split_pointcloud_into_blocks_with_color_claude(pc, block_size=1.0, stride=0.5, block_points=4096, padding=0.001):
    """
    Splits a colored point cloud (XYZ + RGB) into blocks following the document implementation style.
    
    Args:
        pc: Point cloud array with shape (N, 6) where columns are [x, y, z, r, g, b]
        block_size: Size of each block
        stride: Stride for sliding window
        block_points: Number of points per block
        padding: Padding around block boundaries
    
    Returns:
        - block_data: shape (num_blocks, block_points, 9) - [centered_xyz, normalized_rgb, normalized_xyz]
        - point_indices: shape (num_blocks, block_points) - indices of points in each block
    """
    points = pc[:, :6]  # Extract XYZ + RGB
    
    # Get coordinate bounds
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    
    # Calculate grid dimensions
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)
    
    # Initialize output arrays
    data_room, index_room = np.array([]), np.array([])
    
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            # Calculate block boundaries
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            
            # Find points within block boundaries (with padding)
            point_idxs = np.where(
                (points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding) & 
                (points[:, 1] >= s_y - padding) & (points[:, 1] <= e_y + padding)
            )[0]
            
            if point_idxs.size == 0:
                continue
            
            # Calculate number of batches needed and total point size
            num_batch = int(np.ceil(point_idxs.size / block_points))
            point_size = int(num_batch * block_points)
            
            # Sample or repeat points to reach target size
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            
            # Extract batch data
            data_batch = points[point_idxs, :].copy()
            
            # Create normalized coordinates (normalized by scene bounds)
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            
            # Center coordinates relative to block center
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            
            # Normalize RGB values to [0, 1]
            data_batch[:, 3:6] /= 255.0
            
            # Concatenate: [centered_xyz, normalized_rgb, normalized_xyz]
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            
            # Stack data
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
    
    # Reshape to final format
    data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    index_room = index_room.reshape((-1, block_points))
    
    return data_room, index_room

def predict_pointcloud_with_color(pc_raw, model, num_classes=13, device='cpu'):
    """
    Run inference on a colored point cloud using PointNet and return per-point labels.
    """
    model.eval()
    with torch.no_grad():
        block_data, point_indices = split_pointcloud_into_blocks_with_color_claude(pc_raw)
        vote_pool = np.zeros((pc_raw.shape[0], num_classes))

        for i in range(block_data.shape[0]):
            input_block = torch.tensor(block_data[i], dtype=torch.float32).unsqueeze(0).to(device)
            input_block = input_block.transpose(2, 1)
            logits, _ = model(input_block)
            preds = logits.argmax(dim=2).squeeze().cpu().numpy()

            for idx, pt_idx in enumerate(point_indices[i]):
                vote_pool[pt_idx, preds[idx]] += 1

        final_labels = np.argmax(vote_pool, axis=1)
        return final_labels

def predict_single_xyz_pointcloud(pc_raw, model, num_classes=13, device='cpu'):
    """
    Run inference on a single XYZ point cloud using PointNet and return per-point labels.
    """
    start = time.time()
    model.eval()
    with torch.no_grad():
        pc_raw = pc_raw[:, :3] - np.mean(pc_raw[:, :3], axis=0)
        input_block = torch.tensor(pc_raw, dtype=torch.float32).unsqueeze(0).to(device)
        input_block = input_block.transpose(2, 1)
        global_descriptor, logits, _ = model(input_block)
        preds = logits.argmax(dim=2).squeeze().cpu().numpy()
        end = time.time()
        return global_descriptor, preds, end - start


    


