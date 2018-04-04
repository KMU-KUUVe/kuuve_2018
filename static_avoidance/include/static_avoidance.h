#include <ros/ros.h>
#include <iostream>
#include <string>

#include <obstacle_detector/Obstacles.h>
#include <geometry_msgs/Point.h>
// #include <ackermann_msgs/AckermannDriveStamped.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <std_msgs/String.h>
// #include <std_msgs/Empty.h>

#define DETECT_DISTANCE 0.9
#define CONSTANT_STEER 1500
#define CONSTANT_VEL 1520
#define OBSTACLE_RADIUS 0.2

using namespace std;

namespace static_avoidance{

class StaticAvoidance{
public:
	StaticAvoidance();
	StaticAvoidance(ros::NodeHandle nh);
	void initSetup();
    void obstacle_cb(const obstacle_detector::Obstacles& data);

private:
	ros::NodeHandle nh_;
	ros::Publisher pub;
	bool turn_left_flag;
	bool turn_right_flag;
	bool return_left_flag;
	bool return_right_flag;
	bool end_flag;
	int sequence;
	vector<int> steer_buffer;
};

} //end namespace
