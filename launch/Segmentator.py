import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    world_dir = get_package_share_directory('robot_worlds')
    pointnet_dir = get_package_share_directory('robot_pointnet')
    robot_description_path = os.path.join(world_dir, 'urdf', 'robot.urdf')
    world_setup = 'iilab'
    rviz_config = os.path.join(pointnet_dir, 'rviz', 'segmentator.rviz')
    map_yaml = os.path.join(world_dir, 'maps', world_setup, world_setup + '.yaml')

    robot_controller = WebotsController(
        robot_name='robot',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    webots = WebotsLauncher(
        world=os.path.join(world_dir, 'worlds' + '/' +  world_setup + '/' +  world_setup + '.wbt'),
    )


    teleop = Node(
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        name="teleop_twist_keyboard",
        output="screen",
        prefix="gnome-terminal --",
    )

    tf_static_lidar3D = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_lidar_broadcaster",
        arguments=["0.13", "0", "0.25", "0", "0", "0", "base_footprint", "lidar3D"]
    )

    tf_static_lidar2D = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_lidar_broadcaster",
        arguments=["0.13", "0", "0.25", "0", "0", "0", "base_footprint", "lidar2D"]
    )

    tf_static_rgbd = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0.1", "0", "0.3", "0", "-0.3", "0", "base_footprint_real", "rgbd"]
    )

    segmentator = Node(
        package = "robot_pointnet",
        executable="pointnet_segmentator",
        name="pointnet_segmentator"
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_yaml}],
        output='screen'
    )
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        parameters=[{'autostart': True, 'node_names': ['map_server']}],
        output='screen'
    )


    return LaunchDescription([
        tf_static_lidar2D,
        tf_static_lidar3D,
        tf_static_rgbd,
        webots,
        segmentator,
        robot_controller,
        teleop,
        rviz,
        #map_server,
        #lifecycle_manager,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])