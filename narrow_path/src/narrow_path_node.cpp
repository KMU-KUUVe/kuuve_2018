#include "narrow_path.h"

int main(int argc, char* argv[]) {
	ros:: init(argc, argv, "narrow_path_node");
	narrow_path::NarrowPath node("narrow_path");

	cout << "node start" << endl;
	//node.run();
  ros::spin();
	return 0;
}
