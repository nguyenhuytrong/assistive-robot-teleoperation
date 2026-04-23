from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cv",
            executable="compressed_node",
            name="compressed_node",
            parameters=[{"jpeg_quality": 80}],  
        ),
        Node(
            package="cv",
            executable="seg_node",
            name="inference_node",
        ),
        Node(
            package="cv",
            executable="poly_node",
            name="poly_node",
        ),
    ])