import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np
from .pointnet_utils import *
import os
import json



class SceneSegmentor(Node):
    def __init__(self):
        super().__init__('scene_segmentor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rbgd_cloud',  # Change to your topic
            self.pointcloud_callback,
            10
        )
        self.publisher = self.create_publisher(PointCloud2, '/segmented_cloud', 10)
        self.get_logger().info("SceneSegmentor node started.")
        self.save = True

        #segmentation_model = get_shape_segmentation_model(num_points=1024, num_classes=5)
        #print(os.path.abspath(__file__))
        #segmentation_model.load_weights("PTNET.weights.h5")

    def pointcloud_callback(self, msg: PointCloud2):
        if not self.save:
            return
        self.save = False
        # Read point cloud dat

        # Read point cloud into numpy array
        self.get_logger().info("Received point cloud message.")
        points = []
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            points.append((x, y, z))

        points = np.array(points)

        np.save("/home/joao/points.npy", points)

        # Build a new list of (x, y, z, rgb)
        colored_points = []
        for x, y, z in points:
            if z < 0.03:
                r, g, b = 0, 255, 0  # Floor -> Green
            elif z < 1.5:
                r, g, b = 0, 0, 255  # Wall -> Blue
            else:
                r, g, b = 255, 0, 0  # Ceiling -> Red

            rgb = (r << 16) | (g << 8) | b
            colored_points.append((x, y, z, rgb))

        # Create output cloud
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        colored_cloud = point_cloud2.create_cloud(
            header,
            fields=[
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1),
            ],
            points=colored_points
        )

        self.publisher.publish(colored_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = SceneSegmentor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
