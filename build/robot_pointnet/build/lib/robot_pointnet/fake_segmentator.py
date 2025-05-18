import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import std_msgs.msg
import numpy as np
import struct

from .KDTree import create_kd_tree
import tf2_ros
import os
import json

color_map = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
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

        self.target_frame = "map"
        self.pcl_frame = "base_footprint_real"
        self.lidar_tf_buffer = tf2_ros.Buffer()
        self.lidar_tf_listener = tf2_ros.TransformListener(self.lidar_tf_buffer, self)

    def create_rgb_pcl(points, color, frame_id):
        rgb_packed = struct.unpack('I', struct.pack('BBBB', int(color[2]), int(color[1]), int(color[0]), 0))[0]
        cloud_points = []
        for p in points:
            cloud_points.append([p[0], p[1], p[2], rgb_packed])
        
        header = std_msgs.msg.Header()
        header.stamp = rclpy.clock.Clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset = 0, datatype=PointField.FLOAT32, count = 1),
            PointField(name='y', offset = 4, datatype=PointField.FLOAT32, count = 1),
            PointField(name='z', offset = 8, datatype=PointField.FLOAT32, count = 1),
            PointField(name='rgb', offset = 12, datatype=PointField.FLOAT32, count = 1)
        ]

        rgb_pcl = point_cloud2.create_cloud(header, fields, points)
        return rgb_pcl

    def pointcloud_callback(self, msg: PointCloud2):
        self.get_logger().info("x")
        segmented_pcl = PointCloud2()

        try:
            lidar_to_map_tf = self.lidar_tf_buffer.lookup_transform(self.target_frame, self.pcl_frame, rclpy.time.Time())
            self.get_logger().info("Encontrou tf!!!")
            self.get_logger().info("Nao")
            pcl_map = do_transform_cloud(msg, lidar_to_map_tf)
            np_points_map = point_cloud2.read_points_numpy(pcl_map, field_names=("x", "y", "z"), skip_nans=True)
            self.get_logger().info(f"Shape da pcl{np_points_map.shape}")
            inds = np.array([self.kd_tree.query(point, k=1)[1] for point in np_points_map])
            self.get_logger().info(f"Shape dos inds {inds.shape}")
            colors = color_map[inds]
            self.get_logger().info(f"shape da colors {colors.shape}")
            rgb_pcl = self.create_rgb_pcl(np_points_map, colors)
            self.publisher.publish(rgb_pcl)
            return
        except:
            self.get_logger().info("uiui filho, não existe o tf lidar!!")
            return

        #np_points_lidar = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        


def main(args=None):
    rclpy.init(args=args)
    node = SceneSegmentor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
