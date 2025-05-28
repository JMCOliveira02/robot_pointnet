from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_pointnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz*'))),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.[pxy][yma]*'))),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joao',
    maintainer_email='Joao@x.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointnet = robot_pointnet.pointnet:main',
            'fake_segmentator = robot_pointnet.fake_segmentator:main'
        ],
    },
)
