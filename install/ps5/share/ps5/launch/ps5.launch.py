from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Khai báo node joy_node để đọc dữ liệu từ tay cầm PS5
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }]
    )

    # Khai báo node teleop_twist_joy để chuyển tín hiệu joy sang cmd_vel
    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[{
            'require_enable_button': True,
            'enable_button': 10,             # R1
            'axis_linear.x': 1,              # Cần trái tiến/lùi
            'scale_linear.x': 0.6,           # Tốc độ tiến
            'axis_angular.yaw': 0,           # Cần trái xoay
            'scale_angular.yaw': 1.2,        # Tốc độ xoay
        }],
        remappings=[
            ('/cmd_vel', '/controller/cmd_vel') # Remap topic cho đúng với ROSOrin
        ]
    )

#    ps5_haptic_node = Node(
 #       package='ps5',
  #      executable='ps5_haptic',
   #     name='ps5_haptic_node',
    #    output='screen'
    #)

    return LaunchDescription([
        joy_node,
        teleop_node,
        #ps5_haptic_node
    ])