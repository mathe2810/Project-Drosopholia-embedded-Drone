from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='flydrones_ros2',
            executable='flydrones_node',
            name='flydrones_node',
            output='screen',
            parameters=[
                {"brain": "minifly"},
                {"control_hz": 20.0},
                {"config": ""},
            ],
        ),
    ])