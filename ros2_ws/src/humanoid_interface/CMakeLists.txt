cmake_minimum_required(VERSION 3.8)
project(humanoid_interface)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteBehavior.action"
)

install(DIRECTORY humanoid_interface/
  DESTINATION "${PYTHON_INSTALL_DIR}/humanoid_interface"
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/behavior_action_server.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
