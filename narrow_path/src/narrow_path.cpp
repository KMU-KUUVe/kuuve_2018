#include "narrow_path.h"

using namespace std;

namespace narrow_path{

NarrowPath::NarrowPath(std::string name):as_(nh_, name, boost::bind(&NarrowPath::goal_cb, this, _1), false), nh_("~"){
	initSetup();
}

NarrowPath::NarrowPath(std::string name, ros::NodeHandle nh):as_(nh_, name, boost::bind(&NarrowPath::goal_cb, this, _1), false), nh_(nh){
	initSetup();
}

void NarrowPath::initSetup(){
	pub = nh_.advertise<ackermann_msgs::AckermannDriveStamped> ("ackermann", 100);
	sub = nh_.subscribe("/raw_obstacles", 100, &NarrowPath::obstacle_cb, this);

	nh_.getParam("CONST_SPEED", CONST_SPEED);
	nh_.getParam("CONST_STEER", CONST_STEER);
	nh_.getParam("STEER_WEIGHT", STEER_WEIGHT);

	ROS_INFO("const_speed:%d steer:%d", CONST_SPEED, CONST_STEER);

	steer = CONST_STEER;
	speed = CONST_SPEED;

	mean_point_right_y = 0.0;
	mean_point_left_y = 0.0;
	mean_point_y = 0.0;
	end_flag = false;	
	c.x = 100;

	cout << "init finish" << endl;
}

void NarrowPath::obstacle_cb(const obstacle_detector::Obstacles data){
#ifdef DEBUG
	ROS_INFO("Callback function called");
#endif

	
	nh_.getParam("FILTER_RAVA_RADIUS", FILTER_RAVA_RADIUS);
	nh_.getParam("DETE/CT_DISTANCE",DETECT_DISTANCE );
	rava_circles.clear();
	right_circles.clear();
	left_circles.clear();
	// To filter rava obstacle by radius.
	for(int i = 0; i < data.circles.size(); i++){
		if(data.circles[i].radius > FILTER_RAVA_RADIUS){
			rava_circles.push_back(data.circles[i]);
		}
		if(sqrt(data.circles[i].center.x * data.circles[i].center.x + data.circles[i].center.y * data.circles[i].center.y)  <= DETECT_DISTANCE) {
				c = data.circles[i].center;
		}
	}

	for(int i = 0; i < rava_circles.size(); i++) {
		if(rava_circles[i].center.x > -0.2 && rava_circles[i].center.y < 0){ //Right side
			right_circles.push_back(rava_circles[i]);
		}
		else if(rava_circles[i].center.x > -0.2 && rava_circles[i].center.y > 0){ //Left side
			left_circles.push_back(rava_circles[i]);
		}
	}

	//rubber cone sorting
	if(right_circles.size() > 1) //check vector is empty.
		sort(right_circles.begin(), right_circles.end(), cmp);
	if(left_circles.size() > 1) //check vector is empty.
		sort(left_circles.begin(), left_circles.end(), cmp);

#ifdef DEBUG
	cout << "right" << endl;
	for(int i = 0; i < right_circles.size(); i++){
		cout << right_circles[i].center << endl;
	}
	cout << "left" << endl;
	for(int i = 0; i < left_circles.size(); i++){
		cout << left_circles[i].center << endl;
	}
#endif
}

void NarrowPath::goal_cb(const mission_planner::MissionPlannerGoalConstPtr &goal){
	this->run();
	ROS_INFO("Goal_callback");
}

void NarrowPath::run(){
	ros::Rate r(100);
	nh_.getParam("DETECT_DISTANCE", DETECT_DISTANCE);
	nh_.getParam("CONST_SPEED", CONST_SPEED);
	nh_.getParam("CONST_STEER", CONST_STEER);
	nh_.getParam("STEER_WEIGHT",STEER_WEIGHT );
	ROS_INFO("c.x : %f", c.x);
	while(c.x >= DETECT_DISTANCE && ros::ok()){
		ros::spinOnce();			
		steer = CONST_STEER;
		speed = CONST_SPEED;
		ROS_INFO("approaching the obstacle");
	}

	while(ros::ok()){
#ifdef DEBUG
		ROS_INFO("While entered");
#endif
		ros::spinOnce();

		if(left_circles.size() >= 1 && right_circles.size() >= 1){
			end_flag = false;
			if(right_circles.size() >=2){
				mean_point_right_y = (right_circles[0].center.y + right_circles[1].center.y) / 2; 
			}
			else {
				mean_point_right_y = right_circles[0].center.y; 
			}

			if(left_circles.size() >=2){
				mean_point_left_y = (left_circles[0].center.y + left_circles[1].center.y) / 2; 
			}
			else{ 
				mean_point_left_y = left_circles[0].center.y; 
			}

			mean_point_y = mean_point_right_y + mean_point_left_y;

			steer = (mean_point_y * -STEER_WEIGHT) + CONST_STEER;
			speed = CONST_SPEED;
			if(steer > 27){
				steer = 27;
			}
			if(steer < -27){
				steer = -27;
			}
			msg.drive.steering_angle = steer;
			msg.drive.speed = speed;
			pub.publish(msg);
		}
		else{
			end_flag = true;
			steer = 0;
			speed = 0;
		}

		ROS_INFO("Steer:%d Speed:%d", steer, speed);
		ROS_INFO("end_flag: %d", end_flag);
		r.sleep();
	}
}
}//end namespace
