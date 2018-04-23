#include <ros/ros.h>

//#include <std_msgs/String.h>
#include <obstacle_detector/Obstacles.h>
#include <obstacle_detector/CircleObstacle.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <vector>
//#include <string>

using namespace std;

ros::Publisher pub;

#define CONST_STEER 0;
#define CONST_SPEED 7;

bool cmp(const obstacle_detector::CircleObstacle a, const obstacle_detector::CircleObstacle b){
		return (a.center.x < b.center.x);
}

void obstacle_cb(const obstacle_detector::Obstacles data) {
		// vector init
		vector<obstacle_detector::CircleObstacle> rava_circles;
		vector<obstacle_detector::CircleObstacle> left_circles;
		vector<obstacle_detector::CircleObstacle> right_circles;
		for(int i = 0; i < data.circles.size(); i++){
			if(data.circles[i].radius > 0.2){
				rava_circles.push_back(data.circles[i]);
			}
		}
		for(int i = 0; i < rava_circles.size(); i++) {
			if(data.circles[i].center.x > -0.3 && data.circles[i].center.y < 0){ //Right side
					right_circles.push_back(rava_circles[i]);
			}
			else if(data.circles[i].center.x > -0.3 && data.circles[i].center.y > 0){ //Left side
					left_circles.push_back(rava_circles[i]);
			}
		}
/*
		for(int i = 0; i < data.circles.size(); i++) {
				if(data.circles[i].center.x > -0.3 && data.circles[i].center.y < 0){ //Right side
						right_circles.push_back(data.circles[i]);
				}
				else if(data.circles[i].center.x > -0.3 && data.circles[i].center.y > 0){ //Left side
						left_circles.push_back(data.circles[i]);
				}
		}
*/
		//push circle obstacles at right and left vector.
		/*
		cout << "right" << endl;
		for(int i = 0; i < right_circles.size(); i++){
				cout << right_circles[i].center << endl;
		}
		cout << "left" << endl;
		for(int i = 0; i < left_circles.size(); i++){
				cout << left_circles[i].center << endl;
		}
*/
		if(right_circles.size() > 1) //check vector is empty.
				sort(right_circles.begin(), right_circles.end(), cmp);
		if(left_circles.size() > 1) //check vector is empty.
				sort(left_circles.begin(), left_circles.end(), cmp);

		cout << "right" << endl;
		for(int i = 0; i < right_circles.size(); i++){
				cout << right_circles[i].center << endl;
		}
		cout << "left" << endl;
		for(int i = 0; i < left_circles.size(); i++){
				cout << left_circles[i].center << endl;
		}

		int steer = 0; 
		int speed = 0;

		/*
			 if(left_circles.size() >= 2 && right_circles.size() >= 2){
				double mean_point_right_y = (right_circles[0].center.y + right_circles[1].center.y) / 2 
				double mean_point_left_y = (left_circles[0].center.y + left_circles[1].center.y) / 2 
				double mean_point_y = mean_point_right_y + mean_point_left_y;
				//		double mean_point_y = right_circles[0].center.x + right_circles[1].center.x + left_circles[0].center.x + left_circles[1].center.x; 
		*/
		if(left_circles.size() >= 1 && right_circles.size() >= 1){
				double mean_point_right_y = 0;
				double mean_point_left_y = 0;

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
				double mean_point_y = mean_point_right_y + mean_point_left_y;
	
				steer = int(mean_point_y * -20) + CONST_STEER;
				speed = CONST_SPEED;
				cout << mean_point_y << " " << steer << " " << speed << endl;
		}
		//std_msgs::String msg;
		//msg.data = std::to_string(steer) + "," + std::to_string(speed) + ",";	 
		ackermann_msgs::AckermannDriveStamped msg;
		msg.drive.steering_angle = steer;
		msg.drive.speed = speed;
		pub.publish(msg);
}

int main(int argc, char* argv[]) {
		ros::init(argc, argv, "narrow_path_node");
		ros::NodeHandle nh;

		ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, obstacle_cb);

		pub = nh.advertise<ackermann_msgs::AckermannDriveStamped> ("ackermann", 100);
		ros::spin();
}
