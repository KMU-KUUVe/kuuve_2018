#include "narrow_path.h"

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "narrow_path_node");
	narrow_path::NarrowPath node("narrow_path");

	ROS_INFO("node start");
	//node.run();
  	ros::spin();
	return 0;
}
