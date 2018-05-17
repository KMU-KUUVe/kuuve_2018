#include "narrow_path.h"

using namespace std;

namespace narrow_path{

//NarrowPath::NarrowPath(std::string name):as_(nh_, name, boost::bind(&NarrowPath::goal_cb, this, _1), false), nh_("~"){
NarrowPath::NarrowPath(std::string name):as_(nh_, name, boost::bind(&NarrowPath::goal_cb, this, _1), false){
	nh_ = ros::NodeHandle("~");
	as_.start();	
	ROS_INFO("as start");
	initSetup();
}

NarrowPath::NarrowPath(std::string name, ros::NodeHandle nh):as_(nh_, name, boost::bind(&NarrowPath::goal_cb, this, _1), false), nh_(nh){
	as_.start();	
	initSetup();
}

void NarrowPath::initSetup(){
	nh_.getParam("CONST_SPEED", CONST_SPEED);
	nh_.getParam("CONST_STEER", CONST_STEER);
	nh_.getParam("STEER_WEIGHT", STEER_WEIGHT);
	nh_.getParam("FILTER_RAVA_RADIUS", FILTER_RAVA_RADIUS);
	nh_.getParam("DETECT_DISTANCE",DETECT_DISTANCE );
	nh_.getParam("DETECT_DISTANCE", DETECT_DISTANCE);
	nh_.getParam("ONE_SIDE_WEIGHT", one_side_weight);

	ROS_INFO("const_speed:%d steer:%d", CONST_SPEED, CONST_STEER);

	steer = CONST_STEER;
	speed = CONST_SPEED;

	mean_point_right_y = 0.0;
	mean_point_left_y = 0.0;
	mean_point_y = 0.0;
	end_flag = false;	
	end_count = 0;
	c.x = 100;

	ROS_INFO("init finish");
}

void NarrowPath::obstacle_cb(const obstacle_detector::Obstacles data){
#ifdef DEBUG
	ROS_INFO("Callback function called");
#endif
	rava_circles.clear();
	right_circles.clear();
	left_circles.clear();
	// To filter rava obstacle by radius.
	for(int i = 0; i < data.circles.size(); i++){
		if(data.circles[i].radius > FILTER_RAVA_RADIUS){
			rava_circles.push_back(data.circles[i]);
		}
		if((data.circles[i].radius > FILTER_RAVA_RADIUS) && sqrt(data.circles[i].center.x * data.circles[i].center.x + data.circles[i].center.y * data.circles[i].center.y)  <= DETECT_DISTANCE) {
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
	ROS_INFO("Goal_callback");
	pub = nh_.advertise<ackermann_msgs::AckermannDriveStamped> ("/ackermann", 100);
	sub = nh_.subscribe("/raw_obstacles", 100, &NarrowPath::obstacle_cb, this);
	this->run();
}

void NarrowPath::run(){
	ros::Rate r(100);
	ROS_INFO("c.x : %f", c.x);
	while(c.x >= DETECT_DISTANCE && ros::ok()){
		ros::spinOnce();			
		steer = CONST_STEER;
		speed = CONST_SPEED;
		msg.drive.steering_angle = steer;
		msg.drive.speed = speed;
		pub.publish(msg);
		ROS_INFO("approaching the obstacle");
	}

	while(ros::ok()){
#ifdef DEBUG
		ROS_INFO("While entered");
#endif
		ros::spinOnce();

		if(left_circles.size() >= 1 && right_circles.size() >= 1){
			end_flag = false;
			end_count = 0;
			
			if (abs(right_circles[0].center.x - left_circles[0].center.x) > 1){
				mean_point_y = 0;
				one_side_gradient = (right_circles[1].center.y - right_circles[0].center.y)/(right_circles[1].center.x - right_circles[0].center.x);
				if(right_circles[0].center.x < 0.7){
					steer = one_side_gradient * int(one_side_weight);
					ROS_INFO("right obstacles near.%f, %d", one_side_weight, steer);
				}
				else{
					ROS_INFO("right obstacles far.");
					steer = 0;
				}
				speed = CONST_SPEED;
				msg.drive.steering_angle = steer;
				msg.drive.speed = speed;
				ROS_INFO("one_side detect");
			}
			else{
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
			}

			speed = CONST_SPEED;
			if(steer > 26){
				steer = 26;
			}
			if(steer < -26){
				steer = -26;
			}
			msg.drive.steering_angle = steer;
			msg.drive.speed = speed;
			pub.publish(msg);
		}
		else{
			end_count++;
			ROS_INFO("end_count: %d", end_count);
			if(end_count > 100){
				end_flag = true;
				ROS_INFO("end_flag: %d", end_flag);
				steer = 0;
				speed = 0;
				msg.drive.steering_angle = steer;
				msg.drive.speed = speed;
				pub.publish(msg);
				ROS_INFO("Narrow finish");
				as_.setSucceeded(result_);
				break;
			}
		}

		ROS_INFO("Steer:%d Speed:%d", steer, speed);
		ROS_INFO("end_flag: %d", end_flag);
		r.sleep();
	}
}
}//end namespace
