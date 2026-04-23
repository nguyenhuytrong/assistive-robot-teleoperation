from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
    # Read data from ps5
    joy_node1 = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.15,
            'autorepeat_rate': 50.0,
        }],
        remappings=[
            ('/joy', '/ps5/joy')
        ]
    )

    # Transform msg from joy to cmd_vel
    ps5_control_node = Node(
        package='ps5',
        executable='ps5_control_node',
        name='ps5_control_node',
    )

    # Haptic feedback
    ps5_haptic_node = Node(
        package='ps5',
        executable='ps5_haptic',
        name='ps5_haptic_node',
        output='screen',
    )

    return LaunchDescription([
        joy_node1,
        ps5_control_node,
        ps5_haptic_node,
    ])