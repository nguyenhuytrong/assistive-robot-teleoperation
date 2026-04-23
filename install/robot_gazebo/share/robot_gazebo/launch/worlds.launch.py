import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription,LaunchService
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import xacro



def launch_setup(context):
    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    world_name = LaunchConfiguration('world_name', default='empty').perform(context)
    nav = LaunchConfiguration('nav', default='false').perform(context)
    moveit_unite = LaunchConfiguration('moveit_unite', default='false').perform(context)


    moveit_unite_arg = DeclareLaunchArgument('moveit_unite', default_value=moveit_unite)
    nav_arg = DeclareLaunchArgument('nav',default_value=nav)
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time',default_value=use_sim_time)
    world_name_arg = DeclareLaunchArgument('world_name',default_value=world_name)


    # get the package directory
    # pkg_share_dir = get_package_share_directory('holonomic_sim')
    robot_gazebo_path = get_package_share_directory('robot_gazebo')


    # world
    world = os.path.join(robot_gazebo_path,"worlds", world_name+".sdf")
    ign_gz = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory('ros_ign_gazebo'),
                'launch', 'ign_gazebo.launch.py')]),
                launch_arguments=[('ign_args', [' -r ' + world])])

    # ros_ign_bridge
    ros_ign_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_gazebo_path, 'launch/ros_ign_bridge.launch.py')
            ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # spwan_model
    spwan_model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_gazebo_path, 'launch/spwan_model.launch.py')
            ),
        launch_arguments={
            'moveit_unite': moveit_unite,
            'world_name': world_name,
            'use_sim_time': use_sim_time,
        }.items(),
    )


    return ([
        use_sim_time_arg,
        world_name_arg,
        nav_arg,
        moveit_unite_arg,
        ign_gz,
        spwan_model_launch,
        ros_ign_bridge_launch,       
    ])
    


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])



if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
