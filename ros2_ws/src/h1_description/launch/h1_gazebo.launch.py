import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# gz-sim native JointPositionController plugins (see h1.urdf's per-joint
# <topic> overrides) listen on /model/h1/joint/<name>/cmd_pos; bridge each
# ROS -> GZ one-way. (Without the <topic> override, the plugin defaults to
# .../joint/<name>/0/cmd_pos, whose "0" segment is a valid gz-transport name
# but an invalid ROS 2 one -- a name token can't start with a digit. That
# broke both the bridge, which SIGABRTs trying to open it, and any rclpy
# node publishing to it directly, which raises InvalidTopicNameException at
# startup.)
CONTROLLED_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]


def generate_launch_description():
    pkg_h1_description = get_package_share_directory('h1_description')

    # h1.urdf's meshes use package://h1_description/meshes/... URIs, which
    # sdformat_urdf turns into model://h1_description/... -- gz-sim only
    # resolves those against GZ_SIM_RESOURCE_PATH, which nothing sets by
    # default (unlike bipedal_robot_description, h1 actually has meshes).
    # This is cosmetic only (collision geometry is separate primitive
    # shapes), but without it every visual fails to load.
    resource_path = os.path.dirname(pkg_h1_description)
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.pathsep.join(p for p in [resource_path, existing_resource_path] if p)
    )

    # Gazebo Transport pub/sub is scoped by GZ_PARTITION, independent of
    # ROS_DOMAIN_ID (which only isolates the ROS 2/DDS bridge side). Without
    # this, every gz-sim instance on the machine defaults to the same
    # partition and cross-talks -- e.g. this launch's /clock bridge picking
    # up another concurrent gz-sim's clock, which surfaces in ROS as
    # robot_state_publisher's "Moved backwards in time" warning. Default to
    # a partition unique to this process unless the caller already set one.
    gz_partition = SetEnvironmentVariable(
        'GZ_PARTITION',
        os.environ.get('GZ_PARTITION', f'h1-{os.getpid()}')
    )

    z_pos_arg = DeclareLaunchArgument(
        'z_pos',
        default_value='1.05',
        description='Spawn height (m); H1 legs are ~1.0m pelvis-to-sole, straight-legged'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gz-sim server-only (no GUI)'
    )

    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='stand',
        description='Joint commander: stand | walk | behavior | none '
                    '(walk = unitree_rl_gym pretrained H1 policy, '
                    'behavior = humanoid_interface action server)'
    )

    urdf_file = os.path.join(pkg_h1_description, 'urdf', 'h1.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    # Custom world (empty.sdf + the Sensors system plugin the mid360_lidar
    # sensor needs; ros_gz_sim's stock empty.sdf doesn't load it).
    world_file = os.path.join(pkg_h1_description, 'worlds', 'h1_lidar_world.sdf')

    def _gz_sim(context, *args, **kwargs):
        headless = LaunchConfiguration('headless').perform(context).lower() in (
            '1', 'true', 'yes', 'on'
        )
        gz_args = f'-r -s {world_file}' if headless else f'-r {world_file}'
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        get_package_share_directory('ros_gz_sim'),
                        'launch',
                        'gz_sim.launch.py'
                    ])
                ]),
                launch_arguments={'gz_args': gz_args}.items()
            )
        ]

    def _controller(context, *args, **kwargs):
        mode = LaunchConfiguration('controller').perform(context).strip().lower()
        if mode in ('', 'none', 'off'):
            return []
        if mode in ('behavior', 'interface', 'voice'):
            return [
                TimerAction(
                    period=0.2,
                    actions=[
                        Node(
                            package='humanoid_interface',
                            executable='behavior_action_server.py',
                            name='behavior_action_server',
                            parameters=[{'use_sim_time': False}],
                            output='screen',
                            additional_env={
                                'UNITREE_RL_GYM_ROOT': os.environ.get(
                                    'UNITREE_RL_GYM_ROOT',
                                    '/home/asimov/rl-bipedal-walking/unitree_rl_gym',
                                )
                            },
                        ),
                    ],
                )
            ]
        if mode in ('walk', 'rl', 'rl_walk'):
            return [
                # As short a delay as possible: every extra second here is a
                # second the robot's legs have no active target at all
                # (JointPositionController's internal default), which is its
                # own way to fall over before the walk publisher even starts.
                TimerAction(
                    period=0.2,
                    actions=[
                        Node(
                            package='h1_description',
                            executable='h1_rl_walk_publisher.py',
                            name='h1_rl_walk_publisher',
                            parameters=[{'use_sim_time': False}],
                            output='screen',
                            additional_env={
                                'UNITREE_RL_GYM_ROOT': os.environ.get(
                                    'UNITREE_RL_GYM_ROOT',
                                    '/home/asimov/rl-bipedal-walking/unitree_rl_gym',
                                )
                            },
                        ),
                    ],
                )
            ]
        # default: stand
        return [
            TimerAction(
                period=0.5,
                actions=[
                    Node(
                        package='h1_description',
                        executable='standing_pose_publisher.py',
                        name='standing_pose_publisher',
                        parameters=[{'use_sim_time': False}],
                        output='screen',
                    ),
                ],
            )
        ]

    gz_sim = OpaqueFunction(function=_gz_sim)
    controller = OpaqueFunction(function=_controller)

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_h1',
        arguments=[
            '-name', 'h1',
            '-topic', 'robot_description',
            '-z', LaunchConfiguration('z_pos'),
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/empty/model/h1/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            # OdometryPublisher with <odom_topic>odometry</odom_topic> lands on
            # global /odometry in this gz-sim; remap to the ROS name the
            # standing stabilizer subscribes to.
            '/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ] + [
            f'/model/h1/joint/{name}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double'
            for name in CONTROLLED_JOINTS
        ],
        remappings=[
            ('/points/points', '/points'),
            ('/odometry', '/model/h1/odometry'),
            # robot_state_publisher subscribes to /joint_states by default;
            # without this remap it never receives a single update, so it
            # never publishes TF for any revolute joint (torso_joint
            # included) -- breaking the pelvis->torso_link->mid360_link
            # chain RTAB-Map needs to place the lidar cloud ("Could not
            # convert 3d laser scan msg" / TF not set).
            ('/world/empty/model/h1/joint_state', '/joint_states'),
        ],
        output='screen'
    )

    return LaunchDescription([
        gz_resource_path,
        gz_partition,
        z_pos_arg,
        use_sim_time_arg,
        headless_arg,
        controller_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        controller,
    ])
