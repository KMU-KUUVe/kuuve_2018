#include "narrow_path.h"
#include <actionlib/server/simple_action_server.h>
#include <action_with_smach/MissionPlannerAction.h>

int main(int argc, char* argv[]) {
	ros:: init(argc, argv, "narrow_path_node");

	ros::NodeHandle nh_;
	actionlib::SimpleActionServer<action_with_smach::MissionPlannerAction> as_;
	
	//as_.waitForClient

	narrow_path::NarrowPath node(nh_);

	cout << "node start" << endl;
	node.run();
	//ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, obstacle_cb);
	//pub = nh.advertise<ackermann_msgs::AckermannDriveStamped> ("ackermann", 100);
	//pub = nh.advertise<std_msgs::String>("write", 1000);
	//ros::spin();
}
