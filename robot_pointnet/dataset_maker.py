import rclpy
from rclpy.node import Node
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import yaml
import numpy as np
from PIL import Image
from sensor_msgs.msg import PointCloud2 
import time
from sensor_msgs_py import point_cloud2 as pc2

def get_free_world_coordinates(map_yaml_path, map_pgm_path):
    with open(map_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    resolution = config['resolution']
    origin = config['origin']
    negate = config.get('negate', 0)
    
    # Load PGM image
    pgm_image = Image.open(map_pgm_path)
    pgm_data = np.array(pgm_image)
    image_height = pgm_data.shape[0]  # Get image height
    
    # Find free space pixels
    if negate == 0:
        free_mask = pgm_data >= 254
    else:
        free_mask = pgm_data <= 1
    
    # Get pixel coordinates of free space
    free_y, free_x = np.where(free_mask)
    
    # Convert pixel coordinates to world coordinates (matching C++ logic)
    world_x = origin[0] + (free_x + 0.5) * resolution
    world_y = origin[1] + (image_height - free_y - 0.5) * resolution
    
    # Stack into (N, 2) array
    world_coordinates = np.column_stack((world_x, world_y))
    return world_coordinates

def simple_sample(free_coords, target_samples=10000):
    total_coords = len(free_coords)
    if total_coords <= target_samples:
        return free_coords
    
    # Create a grid and sample from each cell
    x_min, x_max = free_coords[:, 0].min(), free_coords[:, 0].max()
    y_min, y_max = free_coords[:, 1].min(), free_coords[:, 1].max()
    
    # Determine grid size to get approximately target_samples
    grid_size = int(np.sqrt(target_samples))
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    sampled_coords = []
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            # Find points in this grid cell
            mask = ((free_coords[:, 0] >= x_bins[i]) & (free_coords[:, 0] < x_bins[i+1]) &
                    (free_coords[:, 1] >= y_bins[j]) & (free_coords[:, 1] < y_bins[j+1]))
            cell_coords = free_coords[mask]
            
            # Sample one point from this cell if it exists
            if len(cell_coords) > 0:
                idx = np.random.randint(len(cell_coords))
                sampled_coords.append(cell_coords[idx])
    
    return np.array(sampled_coords)

class DatasetMaker(Node):
    def __init__(self):
        super().__init__('dataset_maker')
        self.get_logger().info("DatasetMaker node started.")

        self.pose_publisher = self.create_publisher(
            geometry_msgs.msg.PoseStamped, '/set_position', 10)
        
        self.set_pose_confirmation_ = self.create_subscription(
            std_msgs.msg.Int64,
            '/set_position_confirmation',
            self.set_pose_confirmation_callback,
            10
        )

        self.pointcloud_lidar_subscription = self.create_subscription(
            PointCloud2,
            '/scan3D/point_cloud',
            self.pointcloud_callback_lidar,
            10
        )

        self.pointcloud_rgbd_subscription = self.create_subscription(
            PointCloud2,
            '/rgbd_cloud',
            self.pointcloud_callback_rgbd,
            10
        )
        

        self.pcl_count = 0

        free_world_coordinates = get_free_world_coordinates(
            "/home/joao/ros2_ws/src/robot_worlds/maps/iilab/iilab.yaml",
            "/home/joao/ros2_ws/src/robot_worlds/maps/iilab/iilab.pgm"
        )
        self.num_poses = 2000
        self.sampled_coordinates = simple_sample(free_world_coordinates, target_samples=self.num_poses)

        self.get_logger().info(f"Sampled {len(self.sampled_coordinates)} coordinates from the free space.")
        self.most_recent_pcl = None
        self.most_recent_rgbd = None
        self.last_pose_sent = None
        self.publish_pose()  # Publish the first pose immediately
        self.get_logger().info("DatasetMaker node initialized and first pose published.")



        
    
    def publish_pose(self):
        x_sample, y_sample = self.sampled_coordinates[self.pcl_count]
        pose_msg = PoseStamped()
        
        # Set header
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  # or your desired frame
        
        # Set position (example coordinates)
        pose_msg.pose.position.x = x_sample
        pose_msg.pose.position.y = y_sample
        pose_msg.pose.position.z = 0.396
        
        # Set orientation (axis-angle)
        pose_msg.pose.orientation.x = 1.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 0.0

        self.last_pose_sent = np.array([[
            pose_msg.pose.position.x,
            pose_msg.pose.position.y, 
            pose_msg.pose.position.z,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ]])
        
        # Publish the pose
        self.pose_publisher.publish(pose_msg)
        self.pcl_count += 1
        self.get_logger().info(f"Published pose no {self.pcl_count}")

    def pointcloud_callback_lidar(self, msg):
        #self.get_logger().info(f"Received point cloud ")
        self.most_recent_pcl = msg

    def pointcloud_callback_rgbd(self, msg):
        #self.get_logger().info(f"Received RGBD point cloud ")
        self.most_recent_rgbd = msg
    
    def set_pose_confirmation_callback(self, msg):
        self.get_logger().info(f"Received set pose confirmation: {msg.data}")

        if msg.data == self.pcl_count:
            self.get_logger().info(f"Pose {self.pcl_count} confirmed.")
            
            self.timer = self.create_timer(0.5, self.delayed_save_data)

            if self.pcl_count >= self.num_poses: 
                self.get_logger().info("All poses confirmed. Stopping the timer.")
                self.destroy_subscription(self.set_pose_confirmation_)
                self.destroy_subscription(self.pointcloud_lidar_subscription)
                self.destroy_subscription(self.pointcloud_rgbd_subscription)
                self.get_logger().info("Dataset creation completed.")
                self.get_logger().info("Shutting down DatasetMaker node.")
                self.destroy_node()
                rclpy.shutdown()
                return
        else:
            self.get_logger().warn(f"Pose confirmation mismatch: expected {self.pcl_count}, got {msg.data}.")
        # Here you can handle the confirmation message as needed
    
    def delayed_save_data(self):
        # This runs after 0.5 seconds without blocking other callbacks
        lidar_array = np.array(list(pc2.read_points(self.most_recent_pcl, field_names=("x", "y", "z"), skip_nans=True)))
        rgbd_array = np.array(list(pc2.read_points(self.most_recent_rgbd, field_names=("x", "y", "z", "rgb"), skip_nans=True)))
        np.save(f"/home/joao/dev/datasets/iilab_2/lidar/lidar_{self.pcl_count}.npy", lidar_array)
        np.save(f"/home/joao/dev/datasets/iilab_2/poses/pose_{self.pcl_count}.npy", self.last_pose_sent)
        np.save(f"/home/joao/dev/datasets/iilab_2/rgbd/rgbd_{self.pcl_count}.npy", rgbd_array)
        
        # Continue with the rest of your logic...
        self.timer.destroy()  # Clean up the timer
        self.publish_pose()  # Publish the next pose

def main(args=None):
    rclpy.init(args=args)
    node = DatasetMaker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
