cmake_minimum_required(VERSION 3.8)
project(flydrones_ros2)

# Package purement Python : pas de compilation CMake necessaire.
# ament_python gere l'installation via setup.py.

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()