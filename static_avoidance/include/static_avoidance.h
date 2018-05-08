#include <ros/ros.h>
#include <iostream>
#include <string>

#include <obstacle_detector/Obstacles.h>
#include <geometry_msgs/Point.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <std_msgs/String.h>
// #include <std_msgs/Empty.h>
using namespace std;

namespace static_avoidance{

class StaticAvoidance{
public:
	StaticAvoidance();
	StaticAvoidance(ros::NodeHandle nh);
	void initSetup();
    void obstacle_cb(const obstacle_detector::Obstacles& data);
	void run();

private:
	ros::NodeHandle nh_;
	ros::Publisher pub;
	ros::Subscriber sub;
	
	int steer;
	int speed;
	bool turn_left_flag;
	bool turn_right_flag;
	bool return_left_flag;
	bool return_right_flag;
	bool end_flag;
	int sequence;
	bool flag;

	int CONST_VEL;
	int CONST_STEER;
	int DETECT_DISTANCE;
	double OBSTACLE_RADIUS;
	double TURN_FACTOR;
	int TURN_WEIGHT;
	int RETURN_WEIGHT;


	vector<int> steer_buffer;

	geometry_msgs::Point c;
	ackermann_msgs::AckermannDriveStamped msg;
};

} //end namespace
