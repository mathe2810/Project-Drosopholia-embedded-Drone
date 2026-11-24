cmake_minimum_required(VERSION 3.8)
project(flydrones_ros2)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# ===== Python package =====
find_package(ament_cmake_python REQUIRED)
ament_cmake_python_install_package(
  WHERE src/flydrones_ros2
  DESTINATION lib/python3/dist-packages
)

# Make Python scripts executable
install(PROGRAMS
  src/flydrones_ros2/node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Resources =====
install(FILES resource/flydrones_ros2
  DESTINATION share/ament_index/resource_index/packages
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy std_msgs geometry_msgs nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()