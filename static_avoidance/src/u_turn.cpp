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
#include <ros/duration.h>

#define DETECT_DISTANCE 0.5
#define CONSTANT_STEER 30
#define CONSTANT_VEL 1520

using namespace std;
int sequence = 0;

//vector<int> steer_buffer;

//bool ros::Duration::sleep() const;

class Uturn{
private:
    ros::NodeHandle nh;
	geometry_msgs::Point nearest_point;
    std_msgs::String msg;
  	ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, &Uturn::obstacle_cb, this);
	ros::Publisher pub = nh.advertise<std_msgs::String>("write", 1000);

	bool wall_detected = false;
	int speed, steer;

	void obstacle_cb(const obstacle_detector::Obstacles data){
		ROS_INFO("Wall Detected");
		for(int i = 0; i < data.circles.size(); i++){
			if(data.circles[i].center.x < DETECT_DISTANCE && abs(data.circles[i].center.y) < 0.5){
				wall_detected = true;
				ROS_INFO("Wall Detected");
			}
		}
	}

public:
	void execute(){
		ROS_INFO("U turn Start\n");
		while(!(this->wall_detected)){
			steer = 1500;
			speed = 1520;    
  			msg.data = std::to_string(steer) + "," + std::to_string(speed) + "," ;
			pub.publish(msg);
		    ROS_INFO("turning\n");
			//wall_detected = !wall_detected;
		}

		ROS_INFO("Detect!!\n");
		steer = 1100;
		speed = 1520;    
  		msg.data = std::to_string(steer) + "," + std::to_string(speed) + "," ;
		pub.publish(msg);
		ros::Duration(3.0).sleep();

		ROS_INFO("right\n");
		steer = 1900;
		speed = 1520;    
  		msg.data = std::to_string(steer) + "," + std::to_string(speed) + "," ;
		pub.publish(msg);
		ros::Duration(0.8).sleep();

		ROS_INFO("finish\n");
		steer = 1500;
		speed = 1500;    
  		msg.data = std::to_string(steer) + "," + std::to_string(speed) + "," ;
		pub.publish(msg);
		ros::Duration(0.5).sleep();

  	//ros::spin();
	}
};
/*
void calculator(const obstacle_detector::Obstacles data) {
  geometry_msgs::Point c, mycar;
  //bool flag = false;
  printf("go");
  int speed = CONSTANT_VEL;
  int steer = 1500;


  // Select nearest point and assign it to 'c'
   if(abs(data.circles[0].center.y) < 0.5 && data.circles[0].center.x < DETECT_DISTANCE) {
	printf("start u_turn\n");	
	steer = 1100;
	speed = 1520;    
	ros::Duration(3.0).sleep();
	//sleep(3);    
   }
 
 // ackermann_msgs::AckermannDriveStamped msg;
  std_msgs::String msg;
  ROS_INFO("Steer : %d", steer);
  ROS_INFO("Speed : %d", speed);
  ROS_INFO("-----------------------------------------");
    
  msg.data = std::to_string(steer) + "," + std::to_string(speed) + "," ;
  
   pub.publish(msg);
}
*/
int main(int argc, char* argv[]) {
  ros::init(argc, argv, "Static_avoidance_node");

  Uturn uturn;
  uturn.execute();

  //ros::spin();
}
