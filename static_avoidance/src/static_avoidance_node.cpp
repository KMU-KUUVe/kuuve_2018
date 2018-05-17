#include "static_avoidance.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "Static_avoidance_node");
  static_avoidance::StaticAvoidance node("static_avoidance");

  ROS_INFO("node start");
  //node.run();
  ros::spin();
  return 0;
}
