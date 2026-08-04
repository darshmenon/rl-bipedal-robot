import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bipedal_description = get_package_share_directory('bipedal_robot_description')

    # Launch arguments
    x_pos_arg = DeclareLaunchArgument(
        'x_pos',
        default_value='0.0',
        description='X position of the robot'
    )

    y_pos_arg = DeclareLaunchArgument(
        'y_pos',
        default_value='0.0',
        description='Y position of the robot'
    )

    z_pos_arg = DeclareLaunchArgument(
        'z_pos',
        default_value='1.5',
        description='Z position of the robot'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Robot description
    urdf_file = os.path.join(pkg_bipedal_description, 'urdf', 'bipedal.urdf')
    with open(urdf_file, 'r') as f:
        # bipedal.urdf is plain (non-xacro) XML, so the $(find ...) the
        # gz_ros2_control plugin block uses to locate ros2_controllers.yaml
        # needs to be resolved here instead of by a xacro processor.
        robot_description = f.read().replace(
            '$(find bipedal_robot_description)', pkg_bipedal_description)

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    # Launch Gazebo Sim
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # Spawn robot in Gazebo Sim using ros_gz_sim create
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_bipedal_robot',
        arguments=[
            '-name', 'bipedal_robot',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x_pos'),
            '-y', LaunchConfiguration('y_pos'),
            '-z', LaunchConfiguration('z_pos')
        ],
        output='screen'
    )

    # ROS-GZ bridge for topics
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/empty/model/bipedal_robot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        output='screen'
    )

    # gz_ros2_control spawns its own controller_manager inside Gazebo once the
    # robot is created; the spawners need to wait for that to come up.
    controller_spawners = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['leg_position_controller'],
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        x_pos_arg,
        y_pos_arg,
        z_pos_arg,
        use_sim_time_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        controller_spawners
    ])
