import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/joao/ros2_ws/src/robot_pointnet/install/robot_pointnet'
