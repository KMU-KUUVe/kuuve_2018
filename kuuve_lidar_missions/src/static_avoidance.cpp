#include <ros/ros.h>

#include <obstacle_detector/Obstacles.h>
#include <geometry_msgs/Point.h>
// #include <ackermann_msgs/AckermannDriveStamped.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define DETECT_DISTANCE 2.0
#define CONSTANT_STEER 30
#define CONSTANT_VEL 5

using namespace std;

ros::Publisher pub;

bool turn_left_flag = false;
bool turn_right_flag = false;
bool return_left_flag = false;
bool return_right_flag = false;
int sequence = 0;

vector<int> steer_buffer;

void calculator(const obstacle_detector::Obstacles data) {
  geometry_msgs::Point c, mycar;
  bool flag = false;
  int speed = CONSTANT_VEL;
  int steer = 0;


  // Select nearest point and assign it to 'c'
  for(int i = 0; i < data.circles.size(); i++) {
    if(sqrt(data.circles[i].center.x * data.circles[i].center.x + data.circles[i].center.y * data.circles[i].center.y)  <= DETECT_DISTANCE) {
      if(!flag || sqrt(c.x*c.x + c.y*c.y) > sqrt(data.circles[i].center.x * data.circles[i].center.x + data.circles[i].center.y * data.circles[i].center.y)) {
        flag = true;
        c = data.circles[i].center;
      }
    }
  }
  

  double distance = sqrt(c.x * c.x + c.y * c.y);
  if(sequence == 0) { 
    if(flag == 0) {
      if(turn_left_flag) {
        return_right_flag = true;
	turn_left_flag = false;
      }
      else if(return_right_flag) {
  	if(!steer_buffer.empty()) {
          steer = -steer_buffer.back();
	  steer_buffer.pop_back();
	}
	else {
	  return_right_flag = false;
	  sequence++;
 	}
      }  
      else {
        steer = 2;
      }
    }
    else if(flag == 1) {
      if(turn_left_flag) {
	steer = -CONSTANT_STEER / (distance + 1);
        steer_buffer.push_back(steer);
	ROS_INFO("*****");
      }
      else {
     	turn_left_flag = true;
      }
    }
  }
  else if(sequence == 1) {
    if(flag == 0) {
      if(turn_right_flag) {
	return_left_flag = true;
	turn_right_flag = false;
      }
      else if(return_left_flag) {
	if(!steer_buffer.empty()) {
	  steer = -steer_buffer.back();
	  steer_buffer.pop_back();
	}
	else {
	  return_left_flag = false;
	  sequence++;
	}
      }
      else {
	steer = 7;
      }
    }
    else if(flag == 1) {
      if(turn_right_flag) {
	steer = CONSTANT_STEER / (distance+1.8 );
	steer_buffer.push_back(steer);
      }
      else {
	turn_right_flag = true;
      }
    }
  }
  else {
    steer = 2;
  }
  // ackermann_msgs::AckermannDriveStamped msg;
  ROS_INFO("flag : %d", flag);
  ROS_INFO("left turn flag : %d", turn_left_flag);
  ROS_INFO("right turn flag : %d", turn_right_flag);
  ROS_INFO("return left flag : %d", return_left_flag);
  ROS_INFO("return right flag : %d", return_right_flag);
  ROS_INFO("sequence : %d", sequence);
  ROS_INFO("Steer : %d", steer);
  ROS_INFO("Speed : %d", speed);
  ROS_INFO("-----------------------------------------");
  // msg.drive.steering_angle = steer;
  // msg.drive.speed = speed;
  
  // pub.publish(msg);
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "Static_avoidance_node");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, calculator);
  // pub = nh.advertise<ackermann_msgs::AckermannDriveStamped> ("ackermann", 100);
  ros::spin();
}
