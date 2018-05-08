cmake_minimum_required(VERSION 2.8.3)
project(static_avoidance)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  obstacle_detector
  roscpp
  rospy
  std_msgs
  geometry_msgs
  ackermann_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES static_avoidance 
  CATKIN_DEPENDS obstacle_detector roscpp rospy std_msgs ackermann_msgs
  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
)

add_executable(static_avoidance_node src/static_avoidance.cpp src/static_avoidance_node.cpp)
target_link_libraries(static_avoidance_node ${catkin_LIBRARIES})
