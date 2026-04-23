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

    # Haptic feedback
    ps5_haptic_node = Node(
        package='ps5',
        executable='ps5_haptic',
        name='ps5_haptic_node',
        output='screen',
    )

    sac_node = Node(
        package='sac',
        executable='sac_node',
        name='sac_node',
        output='screen',
        parameters=[{
            'scan_topic': '/scan_raw',
            'joy_topic': '/ps5/joy',        # raw human input
            'output_joy_topic': '/sac/joy', # modified output
        }]
    )


    ps5_sac_node = Node(
        package='sac',
        executable='ps5_sac_node',
        name='ps5_sac_node',
        output='screen',
    )

    return LaunchDescription([
        joy_node1,
        ps5_haptic_node,
        sac_node,
        ps5_sac_node,
    ])