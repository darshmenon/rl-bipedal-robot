import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# gz-sim native JointPositionController plugins (see bipedal.urdf's per-joint
# <topic> overrides) listen on /model/bipedal_robot/joint/<name>/cmd_pos;
# bridge each ROS -> GZ one-way. (Without the <topic> override, the plugin
# defaults to .../joint/<name>/0/cmd_pos, whose "0" segment is a valid
# gz-transport name but an invalid ROS 2 one -- a name token can't start
# with a digit. That broke both the bridge, which SIGABRTs trying to open
# it, and any rclpy node publishing to it directly, which raises
# InvalidTopicNameException at startup.)
CONTROLLED_JOINTS = [
    "left_hip_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_joint", "right_knee_joint", "right_ankle_joint",
]

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
        robot_description = f.read()

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
        ] + [
            f'/model/bipedal_robot/joint/{name}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double'
            for name in CONTROLLED_JOINTS
        ],
        remappings=[
            # robot_state_publisher subscribes to /joint_states by default;
            # without this remap it never receives an update, so it never
            # publishes TF for any revolute joint.
            ('/world/empty/model/bipedal_robot/joint_state', '/joint_states'),
        ],
        output='screen'
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
    ])
