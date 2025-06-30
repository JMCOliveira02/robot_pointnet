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
    [0, 0, 255],    #red is wall
    [255, 0, 0],    #green is ceiling
    [0, 255, 0],    #blue is floor
    [0, 0, 0]       #black is unknown
], dtype=np.uint8)

class SceneSegmentor(Node):
    def __init__(self):
        super().__init__('scene_segmentor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/scan3D/point_cloud',  # Change to your topic
            self.pointcloud_callback,
            #self.pointcloud_save,
            10
        )
        self.publisher = self.create_publisher(PointCloud2, '/segmented_cloud', 10)
        #self.get_logger().info("SceneSegmentor node started.")
        self.save = True

        #segmentation_model = get_shape_segmentation_model(num_points=1024, num_classes=5)
        #print(os.path.abspath(__file__))
        #segmentation_model.load_weights("PTNET.weights.h5")
        self.kd_tree, self.all_labels = create_kd_tree()
        self.pcl_points_lidar=np.zeros((1000, 3))
        self.pcl_points_map=np.zeros((1000, 3))

        self.target_frame = "map"
        self.pcl_frame = "lidar3D"
        self.lidar_tf_buffer = tf2_ros.Buffer()
        self.lidar_tf_listener = tf2_ros.TransformListener(self.lidar_tf_buffer, self)
        self.timer = self.create_timer(1, self.publish_segmented_pcl)

    def create_rgb_pcl(self, points, colors, frame_id):
        cloud_points = []
        for p, c in zip(points, colors):
            # Pack color (BGR) into a single float for PointField
            rgb = struct.unpack('f', struct.pack('I', 
                        (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])))[0]
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
        lidar_to_map_tf = tf2_ros.TransformStamped()
        try:
            lidar_to_map_tf = self.lidar_tf_buffer.lookup_transform(self.target_frame, self.pcl_frame, rclpy.time.Time())
            pcl_map = do_transform_cloud(msg, lidar_to_map_tf)
            # Check that the result is not None
            if pcl_map is None:
                return
            # Check the type
            if not isinstance(pcl_map, PointCloud2):
                return
            # Check if the point cloud has any points
            if pcl_map.width == 0 or pcl_map.height == 0:
                return
            # Optionally log its size

            # Try reading the actual points to confirm usability
            self.pcl_points_lidar = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
            self.pcl_points_map = point_cloud2.read_points_numpy(pcl_map, field_names=("x", "y", "z"), skip_nans=True)
            return
        except Exception as e:
            return

    def publish_segmented_pcl(self):
        time_start = time.time()
        #min = np.min(self.np_points[:, 2])
        distances, indices = self.kd_tree.query(self.pcl_points_map, k=1)
        nearest_labels = self.all_labels[indices.flatten()]
        distance_threshold = 0.4
        nearest_labels[distances.flatten() > distance_threshold] = 3
        colors = color_map[nearest_labels]
        rgb_pcl = self.create_rgb_pcl(self.pcl_points_lidar, colors, "lidar3D")
        duration = time.time() - time_start
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
