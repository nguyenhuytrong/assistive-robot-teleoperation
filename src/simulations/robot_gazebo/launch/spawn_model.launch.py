import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription,LaunchService
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,OpaqueFunction
from launch.actions import RegisterEventHandler,TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration,Command
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node

import xacro

def launch_setup(context):
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    world_name = LaunchConfiguration('world', default='robocup_home').perform(context)
    moveit_unite = LaunchConfiguration('moveit_unite', default='false').perform(context)


    sim_ign = 'false' if moveit_unite == 'true' else 'true'

    world_name_arg = DeclareLaunchArgument('world', default_value=world_name)
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value=use_sim_time)

    use_sim_time = True if use_sim_time == 'true' else False


    robot_gazebo_path = os.path.join(get_package_share_directory('robot_gazebo'))

    xacro_file = os.path.join(robot_gazebo_path, 'urdf', 'robot.gazebo.xacro')
    controller_config_file = os.path.join(robot_gazebo_path, 'config', 'robot_cofig.yaml')

    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': Command( \
                    [
                        'xacro ', os.path.join(xacro_file),
                        ' sim_ign:=', sim_ign
                    ]
                ),
                'use_sim_time': use_sim_time

            }
        ],  
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[
            {
                'source_list': ['/controller_manager/joint_states'],
                'rate': 20.0,
                'use_sim_time': use_sim_time          
            }
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )


    ignition_spawn_entity = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description',
                    '-name', 'rosorin',
                    '-allow_renaming', 'true',
                    '-x', '0',
                    '-y', '0'
                    ],
        parameters=[
            {"use_sim_time": True}],
    )


    return [
        use_sim_time_arg,
        world_name_arg,

        joint_state_publisher_node,
        robot_state_publisher_node,

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=ignition_spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        ignition_spawn_entity,
    ]



def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])


if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()