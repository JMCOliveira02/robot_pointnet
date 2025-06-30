import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs.msg
import numpy as np
import struct
import time
import importlib
import torch
import faiss

#from robot_pointnet.yanx27 import pointnet_sem_seg as MODEL
from robot_pointnet.utils import *

POINTNET_MODEL = 'iilab_xyz'
#POINTNET_MODEL = 'yanx27' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointNetSegmentator(Node):
    def __init__(self):
        super().__init__('pointnet_segmentator')
        #
        self.with_color = False
        self.saving_mode = False
        #
        self.get_logger().info("PointNet Segmentator node started !")
        
        #
        if POINTNET_MODEL == 'yanx27':
            self.blocks = True
            self.num_classes = 13
            num_channel = 9
            checkpoint = torch.load('/home/joao/ros2_ws/src/robot_pointnet/robot_pointnet/yanx27/Checkpoints/pointnet_xyzrgbxyz/best_model.pth', weights_only=False, map_location=device)
        elif POINTNET_MODEL == 'iilab_xyz':
            self.blocks = False
            self.num_classes = 4
            num_channel = 3
            checkpoint = torch.load('/home/joao/ros2_ws/src/robot_pointnet/robot_pointnet/iilab/Checkpoints/xyz/best_model.pth', weights_only=False, map_location=device)
        else:
            self.get_logger().error(f"Unknown POINTNET_MODEL: {POINTNET_MODEL}")
            return
        
        model_name = 'pointnet'
        self.MODEL = importlib.import_module('robot_pointnet.' + model_name)
        self.classifier = self.MODEL.get_model(self.num_classes, num_channel)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.eval()
        self.pc_raw = np.array([])

        #
        global_descriptors = np.load("/home/joao/ros2_ws/src/robot_pointnet/robot_pointnet/iilab/Checkpoints/xyz/prior_map.npy")
        self.index = faiss.IndexFlatL2(1024)
        self.index.add(global_descriptors)
        self.pose_array = np.load("/home/joao/ros2_ws/src/robot_pointnet/robot_pointnet/iilab/Checkpoints/xyz/prior_map_poses.npy")
        self.est_pose_array = np.array([])

        #
        if self.with_color: self.subscription = self.create_subscription(PointCloud2, '/rgbd_cloud', self.pointcloud_callback_rgbd, 1)
        else: self.subscription = self.create_subscription(PointCloud2, '/scan3D/point_cloud', self.pointcloud_callback_lidar, 1)
        
        self.pcl_publisher = self.create_publisher(PointCloud2, '/segmented_cloud', 10)
        self.pose_publisher = self.create_publisher(Marker, '/prior_map_poses', 10)
        
        if self.saving_mode: self.timer_saving = self.create_timer(1, self.save_pointclouds)
        else: 
            self.timer = self.create_timer(1, self.publish_segmented_pcl)


    def create_rgb_pcl(self, points, colors, frame_id, stamp):
        cloud_points = []
        for p, c in zip(points, colors):
            rgb = struct.unpack('f', struct.pack('I', 
                        (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])))[0]
            cloud_points.append([p[0], p[1], p[2], rgb])

        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, cloud_points)

    def pointcloud_callback_rgbd(self, msg):
        #self.get_logger().info("PCL!")
        rgbd_points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        self.pcl_points = rgbd_points[:, :3]
        finite_mask = np.isfinite(self.pcl_points).all(axis=1)
        self.pcl_points = self.pcl_points[finite_mask]
        rgb_packed = rgbd_points[:, 3]
        rgb_packed = rgb_packed[finite_mask]
        pcl_colors = np.array([unpack_rgb(rgb) for rgb in rgb_packed])
        self.pc_raw = np.concatenate([self.pcl_points, pcl_colors], axis=1)
        #self.get_logger().info(f"RGB data type: {rgbd_points[:, 3].dtype}")
        #self.get_logger().info(f"Number of points {len(self.pc_raw)}")
        #self.get_logger().info(f"Number of points {len(self.pc_raw)}")
        return
    
    def pointcloud_callback_lidar(self, msg : PointCloud2):
        #self.get_logger().info("PCL!")
        self.timestamp_pc_raw = msg.header._stamp
        self.pcl_points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        finite_mask = np.isfinite(self.pcl_points).all(axis=1)
        self.pcl_points = self.pcl_points[finite_mask]
        if self.with_color:
            pcl_colors = np.ones_like(self.pcl_points)
            self.pc_raw = np.concatenate([self.pcl_points, pcl_colors], axis=1)
        else:
            self.pc_raw = self.pcl_points
        #self.get_logger().info(f"pc raw size: {len(self.pc_raw)}")
        return


    def publish_segmented_pcl(self):
        #time_start = time.time()
        if len(self.pc_raw) == 0:
            self.get_logger().warn("No point cloud data available")
            return
        if self.blocks: 
            pred_labels = predict_pointcloud_with_color(self.pc_raw, self.classifier, num_classes=self.num_classes, device=device)
        else:
            global_descriptor, pred_labels, duration = predict_single_xyz_pointcloud(self.pc_raw, self.classifier, num_classes=self.num_classes, device=device)
            self.get_prior_map_poses(global_descriptor)   
        #self.get_logger().info(f"Prediction time: {duration:.4f} seconds")
        #self.get_logger().info(f"Global descriptor shape: {global_descriptor.shape}")


        
        colors = np.array([colors_segmentation[label] for label in pred_labels])
        #self.get_logger().info(f"Pred labels: {len(pred_labels)}")
        if self.with_color: frame = "rgbd"
        else: frame = "lidar3D"
        rgb_pcl = self.create_rgb_pcl(self.pcl_points, colors , frame, self.timestamp_pc_raw)
        #duration = time.time() - time_start
        self.pcl_publisher.publish(rgb_pcl)
        self.publish_poses_rviz()
        return 
    
    #Save the pc_raw array to a file
    def save_pointclouds(self):
        if len(self.pc_raw) == 0:
            self.get_logger().warn("No point cloud data available to save")
            return
        if self.with_color: frame = "rgbd"
        else: frame = "lidar3D"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"/home/joao/ros2_ws/src/robot_pointnet/robot_pointnet/Pointclouds/{frame}/pc_raw_{timestamp}.npy"
        np.save(filename, self.pc_raw)
        self.get_logger().info(f"Saved point cloud to {filename}")

    def get_prior_map_poses(self, descriptor):
        k = 5
        distances, indices = self.index.search(descriptor.reshape(1, -1), k)
        self.est_pose_array = [self.pose_array[i] for i in indices[0]]
        self.est_pose_array = np.array(self.est_pose_array)
        #self.get_logger().info(f"Estimated poses: {len(self.est_pose_array)}")
        return

    def publish_poses_rviz(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = float(1)
        marker.scale.x = float(0.3)
        marker.scale.y = float(0.3)
        marker.scale.z = float(0.3)
        marker.color.a = float(1)
        marker.color.r = float(1)
        marker.color.g = float(0)
        marker.color.b = float(0)
        marker.points = []

        for point in self.est_pose_array:
            point_ = Point()
            point_.x = float(point[0])
            point_.y = float(point[1])
            point_.z = float(0.0)
            marker.points.append(point_)
            
        self.pose_publisher.publish(marker)
    


def main(args=None):
    rclpy.init(args=args)
    node = PointNetSegmentator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
