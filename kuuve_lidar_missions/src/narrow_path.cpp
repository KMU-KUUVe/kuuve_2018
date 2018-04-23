#include "narrow_path.h"

using namespace std;

namespace narrow_path{

NarrowPath::NarrowPath():nh_(){
	initSetup();
}

NarrowPath::NarrowPath(ros::NodeHandle nh):nh_(nh){
	initSetup();
}

void NarrowPath::initSetup(){
	pub = nh_.advertise<ackermann_msgs::AckermannDriveStamped> ("ackermann", 100);
	sub = nh_.subscribe("raw_obstacles", 100, &NarrowPath::obstacle_cb, this);

	steer = 0;
	speed = 6;
	mean_point_right_y = 0.0;
	mean_point_left_y = 0.0;
	mean_point_y = 0.0;
	end_flag = false;	

	cout << "init finish" << endl;
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
		//if(data.circles[i].radius > )
		rava_circles.push_back(data.circles[i]);
	}

	for(int i = 0; i < rava_circles.size(); i++) {
		if(rava_circles[i].center.x > -0.2 && rava_circles[i].center.y < 0){ //Right side
			right_circles.push_back(rava_circles[i]);
		}
		else if(rava_circles[i].center.x > -0.2 && rava_circles[i].center.y > 0){ //Left side
			left_circles.push_back(rava_circles[i]);
		}
	}

	/* /////////////////////////////original code/////////////////////////////
	   for(int i = 0; i < data.circles.size(); i++) {
	   if(data.circles[i].center.x > -0.2 && data.circles[i].center.y < 0){ //Right side
	   right_circles.push_back(data.circles[i]);
	   }
	   else if(data.circles[i].center.x > -0.2 && data.circles[i].center.y > 0){ //Left side
	   left_circles.push_back(data.circles[i]);
	   }
	   }

*/

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
	/*
	   ackermann_msgs::AckermannDriveStamped msg;
	   msg.drive.steering_angle = steer;
	   msg.drive.speed = speed;
	   pub.publish(msg);
	   */
}
void calculate_points(){
}
void publish(){
}
void NarrowPath::run(){
	ros::Rate r(100);
	while(ros::ok()){
#ifdef DEBUG
		ROS_INFO("While entered");
#endif
		ros::spinOnce();

		if(left_circles.size() >= 1 && right_circles.size() >= 1){
		
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

			steer = int(mean_point_y * -20) + CONST_STEER;
			speed = CONST_SPEED;

			ROS_INFO("Steer:%d Speed:%d", steer, speed);

			msg.drive.steering_angle = steer;
			msg.drive.speed = speed;
			pub.publish(msg);
		}
		r.sleep();
	}
}
	//std_msgs::String msg;
	//msg.data = std::to_string(steer) + "," + std::to_string(speed) + ",";	 
}//end namespace
