#include "static_avoidance.h"
//#include <actionlib/server/simple_action_server.h>
//#include <action_with_smach/MissionPlannerAction.h>

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "Static_avoidance_node");

  static_avoidance::StaticAvoidance node;

  cout << "node start"<< endl;
  node.run();

  //ros::Subscriber sub = nh.subscribe("raw_obstacles", 1, calculator);
  //pub = nh.advertise<std_msgs::String>("write", 1000);
 // ros::spin();
}
