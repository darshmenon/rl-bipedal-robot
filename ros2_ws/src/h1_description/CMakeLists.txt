cmake_minimum_required(VERSION 3.8)
project(h1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  urdf
  meshes
  config
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/standing_pose_publisher.py
  scripts/gz_pose_to_odom.py
  scripts/h1_rl_walk_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
