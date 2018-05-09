cmake_minimum_required(VERSION 2.8.3)
project(narrow_path)

add_definitions(-std=c++11)

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
  actionlib_msgs
)

generate_messages(
  DEPENDENCIES 
  actionlib_msgs 
  std_msgs
  sensor_msgs
  ackermann_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES narrow_path
  CATKIN_DEPENDS obstacle_detector roscpp rospy std_msgs actionlib_msgs
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

add_executable(narrow_path_node src/narrow_path.cpp src/narrow_path_node.cpp)
#add_executable(narrow_path_action src/narrow_path_action_node.cpp src/narrow_path_action.cpp)
target_link_libraries(narrow_path_node ${catkin_LIBRARIES})
add_dependencies(
  narrow_path_node
  ${narrow_path_EXPORTED_TARGETS}
)
