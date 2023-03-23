cmake_minimum_required(VERSION 3.5)
project(cc_visual_computing)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(libpsaf_msgs REQUIRED)
find_package(libpsaf REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(image_transport REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(sliding_window_src
  src/sliding_window.cpp
  src/sliding_window_node.cpp
)

set(probabilistic_tracking_src
  src/probabilistic_tracking.cpp
  src/probabilistic_tracking_node.cpp
)

add_executable(sliding_window ${sliding_window_src})
target_include_directories(sliding_window PRIVATE "include")
ament_target_dependencies(sliding_window rclcpp std_msgs sensor_msgs libpsaf_msgs libpsaf)

add_executable(probabilistic_tracking ${probabilistic_tracking_src})
target_include_directories(probabilistic_tracking PRIVATE "include" PRIVATE "/usr/include/eigen3")
ament_target_dependencies(probabilistic_tracking rclcpp std_msgs sensor_msgs libpsaf_msgs libpsaf Eigen3)

install(TARGETS
  sliding_window
  probabilistic_tracking
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
