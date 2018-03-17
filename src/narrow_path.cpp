#include <ros/ros.h>

#include <std_msgs/String.h>
#include <obstacle_detector/Obstacles.h>
#include <obstacle_detector/CircleObstacle.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <vector>

using namespace std;

ros::Publisher pub;

double distance_from_car(geometry_msgs::Point p1, geometry_msgs::Point p2){
  double gradient = (p1.x - p2.x) / (p1.y - p2.y);
  double distance = abs(-gradient * p1.y + p1.x) / sqrt(1 + gradient * gradient);
  return distance;
}

bool cmp(obstacle_detector::CircleObstacle a, obstacle_detector::CircleObstacle b){
		return a.center.x < b.center.x;
}

void obstacle_cb(const obstacle_detector::Obstacles data) {
  	// vector init
	vector<obstacle_detector::CircleObstacle> left_circles;
	vector<obstacle_detector::CircleObstacle> right_circles;

	cout << 1 << endl;

	for(int i = 0; i < data.circles.size(); i++) {
		if(data.circles[i].center.x > 0 && data.circles[i].center.y < 0){ //Right side
				right_circles.push_back(data.circles[i]);
		}
		else if(data.circles[i].center.x > 0 && data.circles[i].center.y > 0){ //Left side
				left_circles.push_back(data.circles[i]);
		}
	}
	
	cout << 2 << endl;
	
	//push circle obstacles at right and left vector.
	cout << "right" << endl;
	for(int i = 0; i < right_circles.size(); i++){
	  cout << right_circles[i].center << endl;
	}
	cout << "left" << endl;
	for(int i = 0; i < left_circles.size(); i++){
	  cout << left_circles[i].center << endl;
	}
	
	if(right_circles.size() > 0) //check vector is empty.
		sort(right_circles.begin(), right_circles.end(), cmp);
	if(left_circles.size() > 0) //check vector is empty.
	  sort(left_circles.begin(), right_circles.end(), cmp);

	cout << 3 << endl;

	for(int i = 0; i < right_circles.size(); i++){
	  cout << right_circles[i].center << endl;
	}
  
 // pub.publish(msg);
}

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "PID_Controller_node");
	ros::NodeHandle nh;

	ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, obstacle_cb);

	pub = nh.advertise<std_msgs::String> ("write", 10);
	ros::spin();
}

