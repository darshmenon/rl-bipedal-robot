from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )

    behavior_server = Node(
        package="humanoid_interface",
        executable="behavior_action_server.py",
        name="behavior_action_server",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        additional_env={
            "UNITREE_RL_GYM_ROOT": "/home/asimov/rl-bipedal-walking/unitree_rl_gym",
        },
    )

    return LaunchDescription([use_sim_time, behavior_server])
