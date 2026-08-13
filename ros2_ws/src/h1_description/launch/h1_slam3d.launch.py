"""3D LiDAR SLAM (RTAB-Map) for the Unitree H1 humanoid.

Scope: H1 is held in a fixed standing pose (see standing_pose_publisher.py --
no walking controller exists yet) while its Mid-360-mount gpu_lidar
(h1.urdf's mid360_lidar sensor) feeds RTAB-Map ICP SLAM, mirroring
quadruped-dog-rl's proven slam3d_go2.launch.py setup for the Go2's lidar3d
sensor. Move the robot by nudging it in the Gazebo GUI (or applying an
external force/teleop later) to grow the map; it won't walk on its own.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch h1_description h1_slam3d.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

RTABMAP_PARAMS = {
    "use_sim_time": True,
    "frame_id": "pelvis",
    "odom_frame_id": "odom",
    "map_frame_id": "map",
    "subscribe_depth": False,
    "subscribe_rgb": False,
    "subscribe_scan_cloud": True,
    "approx_sync": True,
    "wait_for_transform": 0.3,
    # RTAB-Map params are strings.
    "Reg/Strategy": "1",           # ICP -- no camera for Vis registration
    "Icp/PointToPlane": "true",
    "Grid/Sensor": "0",            # occupancy grid from the lidar cloud
    "Grid/3D": "false",            # projected 2D grid
    "Grid/CellSize": "0.05",
    "Grid/RangeMax": "20.0",
    # Same fix as quadruped-dog-rl's lidar3d: a sparse 16-channel vertical
    # resolution sprinkles spurious "obstacle" cells on flat ground.
    "Grid/NoiseFilteringRadius": "0.1",
    "Grid/NoiseFilteringMinNeighbors": "5",
    "Mem/IncrementalMemory": "true",
}


def generate_launch_description():
    h1_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('h1_description'),
                'launch',
                'h1_gazebo.launch.py'
            ])
        ]),
    )

    # H1 isn't walking under its own control yet, so there's no real state
    # estimator -- republish Gazebo's ground-truth model pose as /odom + TF
    # (same technique as quadruped-dog-rl's scripts/gz_pose_to_odom.py).
    ground_truth_to_odom = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='h1_description',
                executable='gz_pose_to_odom.py',
                name='gz_pose_to_odom',
                parameters=[{'use_sim_time': True}],
                output='screen'
            ),
        ]
    )

    # Started after the standing pose has settled and odom has been
    # publishing for a few seconds, so RTAB-Map's TF lookups don't race
    # startup.
    rtabmap_slam = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='rtabmap_slam',
                executable='rtabmap',
                name='rtabmap',
                output='screen',
                parameters=[RTABMAP_PARAMS],
                remappings=[('odom', '/odom'), ('scan_cloud', '/points')],
                arguments=['-d'],  # fresh database each run
            ),
        ]
    )

    return LaunchDescription([
        h1_gazebo,
        ground_truth_to_odom,
        rtabmap_slam,
    ])
