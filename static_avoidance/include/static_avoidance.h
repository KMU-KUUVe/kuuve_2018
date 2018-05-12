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

#include <actionlib/server/simple_action_server.h>
#include <mission_planner/MissionPlanerAction.h>

using namespace std;

namespace static_avoidance{

class StaticAvoidance{
public:
	StaticAvoidance(std::string name);
	StaticAvoidance(std::string name, ros::NodeHandle nh);
	void initSetup();
    	void obstacle_cb(const obstacle_detector::Obstacles& data);
	void goal_cb(const mission_planner::MissionPlannerGoalConstPtr &goal);	
	void run();

private:
	ros::NodeHandle nh_;
	ros::Publisher pub;
	ros::Subscriber sub;
	
	actionlib::SimpleActionServer<mission_planner::MissionPlannerAction> as_;
	mission_planner::MissionPlannerResult result_;
	
	int steer;
	int speed;
	bool turn_left_flag;
	bool turn_right_flag;
	bool return_left_flag;
	bool return_right_flag;
	bool end_flag;
	int sequence;
	bool flag;
	int end_count;

	int CONST_VEL;
	int CONST_STEER;
	int DETECT_DISTANCE;
	double OBSTACLE_RADIUS;
	double TURN_FACTOR;
	int TURN_WEIGHT1;
	int TURN_WEIGHT2;
	int RETURN_WEIGHT1;
	int RETURN_WEIGHT2;


	vector<int> steer_buffer;

	geometry_msgs::Point c;
	ackermann_msgs::AckermannDriveStamped msg;
};

} //end namespace
