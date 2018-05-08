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

ros::Publisher pub;

int sequence = 0;

//vector<int> steer_buffer;

//bool ros::Duration::sleep() const;

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

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "Static_avoidance_node");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, calculator);
  pub = nh.advertise<std_msgs::String>("write", 1000);
  ros::spin();
}
