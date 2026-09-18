#!/usr/bin/env python
import os
import xacro
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot").perform(context)
    ns = LaunchConfiguration("ns").perform(context)
    world_name = LaunchConfiguration("world").perform(context)

    world_file = os.path.join(
        FindPackageShare("gazebo_bridge").find("gazebo_bridge"),
        "worlds",
        world_name,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("gazebo_ros"), "launch"),
                "/gazebo.launch.py",
            ]
        ),
        launch_arguments={
            "world": world_file,
            "pause": "false",
            "verbose": "false",
        }.items(),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", f"{ns}/robot_description",
            "-entity", f"{ns}",
            "-robot_namespace", f"{ns}",
            "-x", "0.", "-y", "0.", "-z", "0.32",
        ],
        output="screen",
    )

    robot_xacro_path = os.path.join(
        get_package_share_directory(robot_name + "_description"),
        "xacro", "robot.xacro",
    )
    robot_description = xacro.process_file(
        robot_xacro_path, mappings={"hw_env": "gazebo"}
    ).toxml()
    # mesh paths: package:// -> file://
    robot_description = robot_description.replace(
        "package://" + robot_name + "_description",
        "file://" + get_package_share_directory(robot_name + "_description"),
    )
    # controllers.yaml: generic gazebo_bridge path -> robot-specific
    robot_description = robot_description.replace(
        get_package_share_directory("gazebo_bridge") + "/config/controllers.yaml",
        get_package_share_directory("rl_controller") + "/config/" + robot_name + "/controllers.yaml",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": True},
            {"publish_frequency": 15.0},
            {"frame_prefix": ns + "/"},
        ],
        namespace=ns,
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager",
                   ns + "/controller_manager"],
    )
    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster", "--controller-manager",
                   ns + "/controller_manager"],
    )
    rl_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[robot_name + "_rl_controller", "--controller-manager",
                   ns + "/controller_manager"],
    )
    return [robot_state_pub_node, gazebo, spawn_entity,
            joint_state_broadcaster_spawner, imu_sensor_broadcaster_spawner,
            rl_controller_spawner]


def generate_launch_description():
    declared_arguments = [
        launch.actions.DeclareLaunchArgument(
            "robot", default_value="dog3",
            description="Robot description package name (without _description suffix)",
        ),
        launch.actions.DeclareLaunchArgument(
            "ns", default_value="",
            description="Namespace",
        ),
        launch.actions.DeclareLaunchArgument(
            "world", default_value="empty_world.world",
            description="World file under gazebo_bridge/worlds (empty_world.world | obstacle_field.world)",
        ),
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
