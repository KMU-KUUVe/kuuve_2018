cmake_minimum_required(VERSION 2.8.3)
project(crosswalk_stop)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1)

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	OpenCV REQUIRED
	roscpp
	std_msgs
	image_transport
	sensor_msgs
	mission_planner
)

#find_package(OpenCV 3 REQUIRED)

#include_directories(${OpenCV_INCLUDE_DIRS})

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

#add_action_files(
#	DIRECTORY action
#	FILES MissionPlanner.action
#)

#generate_messages(
#	DEPENDENCIES actionlib_msgs std_msgs
#)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES opencv_test
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs 
  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS} ./include
)

include_directories(
	${lane_detector_SOURCE_DIR}/include
)

include_directories(
	include ${Boost_INCLUDE_DIRS}
)


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/opencv_test.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide

# add_executable(crosswalk_stop src/main.cpp src/LineDetector.cpp)
add_executable(crosswalk_stop src/main.cpp src/CrosswalkStop.cpp src/CrosswalkStopNode.cpp ${lane_detector_SOURCE_DIR}/src/LinePointDetector.cpp ${lane_detector_SOURCE_DIR}/src/LaneDetector.cpp ${lane_detector_SOURCE_DIR}/src/InToOutLaneDetector.cpp ${lane_detector_SOURCE_DIR}/src/MyException.cpp)
add_dependencies(crosswalk_stop ${crosswalk_stop_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(crosswalk_stop
  ${catkin_LIBRARIES}
)
