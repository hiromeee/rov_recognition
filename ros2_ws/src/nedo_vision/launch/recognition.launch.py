import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('nedo_vision')
    default_config = os.path.join(pkg_share, 'configs', 'camera_hsv_baseline.yaml')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to pipeline YAML config file',
    )
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/mavlink_communicator/current_pose_acc',
        description='IMU topic name',
    )
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera_driver/camera_image',
        description='Camera image topic name',
    )

    node = Node(
        package='nedo_vision',
        executable='recognition_node',
        name='recognition_node',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'camera_topic': LaunchConfiguration('camera_topic'),
        }],
        output='screen',
    )

    return LaunchDescription([
        config_file_arg,
        imu_topic_arg,
        camera_topic_arg,
        node,
    ])
