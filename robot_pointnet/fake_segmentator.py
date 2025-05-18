import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import std_msgs.msg
import numpy as np
import struct
import time

from .KDTree import create_kd_tree
import tf2_ros
import os
import json

color_map = np.array([
    [255, 0, 0],    #red is wall
    [0, 255, 0],    #green is ceiling
    [0, 0, 255],    #blue is floor
    [0, 0, 0]       #black is unknown
], dtype=np.uint8)

class SceneSegmentor(Node):
    def __init__(self):
        super().__init__('scene_segmentor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/scan/point_cloud',  # Change to your topic
            self.pointcloud_callback,
            10
        )
        self.publisher = self.create_publisher(PointCloud2, '/segmented_cloud', 10)
        self.get_logger().info("SceneSegmentor node started.")
        self.save = True

        #segmentation_model = get_shape_segmentation_model(num_points=1024, num_classes=5)
        #print(os.path.abspath(__file__))
        #segmentation_model.load_weights("PTNET.weights.h5")
        self.kd_tree, self.all_labels = create_kd_tree()
        self.np_points=np.zeros((1000, 3))

        self.target_frame = "map"
        self.pcl_frame = "base_footprint_real"
        self.lidar_tf_buffer = tf2_ros.Buffer()
        self.lidar_tf_listener = tf2_ros.TransformListener(self.lidar_tf_buffer, self)
        self.timer = self.create_timer(1, self.publish_segmented_pcl)

    def create_rgb_pcl(self, points, colors, frame_id):
        cloud_points = []
        for p, c in zip(points, colors):
            # Pack color (BGR) into a single float for PointField
            rgb = struct.unpack('f', struct.pack('I', 
                        (int(c[2]) << 16) | (int(c[1]) << 8) | int(c[0])))[0]
            cloud_points.append([p[0], p[1], p[2], rgb])

        header = std_msgs.msg.Header()
        header.stamp = rclpy.clock.Clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, cloud_points)


    def pointcloud_callback(self, msg: PointCloud2):
        self.get_logger().info("x")
        segmented_pcl = PointCloud2()

        self.get_logger().info("x")
        segmented_pcl = PointCloud2()
        lidar_to_map_tf = tf2_ros.TransformStamped()

        try:
            lidar_to_map_tf = self.lidar_tf_buffer.lookup_transform(self.target_frame, self.pcl_frame, rclpy.time.Time())
            pcl_map = do_transform_cloud(msg, lidar_to_map_tf)
            # Check that the result is not None
            if pcl_map is None:
                self.get_logger().error("pcl_map is None after transformation")
                return
            # Check the type
            if not isinstance(pcl_map, PointCloud2):
                self.get_logger().error("pcl_map is not a PointCloud2 message")
                return
            # Check if the point cloud has any points
            if pcl_map.width == 0 or pcl_map.height == 0:
                self.get_logger().warn("Transformed point cloud is empty")
                return
            # Optionally log its size
            self.get_logger().info(f"pcl_map width: {pcl_map.width}, height: {pcl_map.height}")

            # Try reading the actual points to confirm usability
            self.np_points = point_cloud2.read_points_numpy(pcl_map, field_names=("x", "y", "z"), skip_nans=True)
            self.get_logger().info(f"Valid transformed point cloud with {self.np_points.shape[0]} points")
            return
        except Exception as e:
            self.get_logger().error(f"Error during point cloud transform or validation: {e}")
            return
    def pointcloud_save(self, msg:PointCloud2):
        return
        np_points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        np.save("/home/joao/ros2_ws/src/robot_pointnet/lidar_reading.npy", np_points)
        self.get_logger().info("Saved Pointcloud")

    def publish_segmented_pcl(self):
        #inds = np.zeros(len(np_points), dtype=np.uint8)
        time_start = time.time()
        distances, indices = self.kd_tree.query(self.np_points, k=1)
        nearest_labels = self.all_labels[indices.flatten()]
        distance_threshold = 0.5
        nearest_labels[distances.flatten() > distance_threshold] = 3
        colors = color_map[nearest_labels]
        self.get_logger().info(f"shape da colors {colors.shape}")
        rgb_pcl = self.create_rgb_pcl(self.np_points, colors, "map")
        duration = time.time() - time_start
        self.get_logger().info(f"Query and pcl creation took {duration:.6f} seconds")
        self.publisher.publish(rgb_pcl)

        return 


def main(args=None):
    rclpy.init(args=args)
    node = SceneSegmentor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
