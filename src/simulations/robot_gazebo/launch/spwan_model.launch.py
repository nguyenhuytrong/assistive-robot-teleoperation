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
    rviz_file = os.path.join(robot_gazebo_path, 'rviz', 'robot_gazebo.rviz')
    controller_config_file = os.path.join(robot_gazebo_path, 'config', 'robot_config.yaml')

    # 生成机器人描述
    robot_description_content = Command([
        'xacro ', os.path.join(xacro_file),
        ' sim_ign:=', sim_ign
    ])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description_content,
                'use_sim_time': use_sim_time
            }
        ],  
    )

    # 添加控制器管理器节点 - 这是关键缺失部分
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description_content},
            controller_config_file
        ],
        output='screen',
        condition=IfCondition(sim_ign)  # 只在仿真时启动
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[
            {
                'source_list': ['/joint_states'],  # 修改为正确的主题
                'rate': 20.0,
                'use_sim_time': use_sim_time          
            }
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    static_transform_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ign_lidar_link_to_lidar_link',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0','robot/base_footprint/gpu_lidar','lidar_link']
    )

    map_odom_nav_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'map', 'odom']
    )

    ignition_spawn_entity = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description',
                    '-name', 'robot',
                    '-allow_renaming', 'true',
                    '-x', '0',
                    '-y', '0',
                    '-z', '0.0'
                    ],
        parameters=[
            {"use_sim_time": True}],
    )

    return [
        use_sim_time_arg,
        world_name_arg,

        # 首先启动控制器管理器
        controller_manager_node,
        
        # 然后启动其他节点
        joint_state_publisher_node,
        robot_state_publisher_node,

        # 在 Ignition 实体生成后启动控制器
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