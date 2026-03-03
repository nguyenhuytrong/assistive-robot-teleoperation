from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
    # Khai báo node joy_node để đọc dữ liệu từ tay cầm PS5
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

    ps5_control_node = Node(
        package='ps5',
        executable='ps5_control_node',
        name='ps5_control_node',
    )

    return LaunchDescription([
        joy_node1,
        ps5_control_node,
        #ps5_haptic_node
    ])