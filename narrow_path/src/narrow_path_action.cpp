#include "narrow_path_action.h"

namespace narrow_path_action{
NarrowPathAction::NarrowPathAction(std::string name){
	name_ = name;
  as_ = SimpleActionServer<mission_planner::MissionPlannerAction>(nh_, name, &NarrowPathAction::action_cb, this);
}
/*
NarrowPathAction::NarrowPathAction(std::string name, ros::NodeHandle nh) : NarrowPath(nh){
	name_ = name;
  as_ = SimpleActionServer<mission_planner::MissionPlannerAction>(nh_, name, &NarrowPathAction::action_cb, this);
}
*/
void NarrowPathAction::action_cb(const mission_planner::MissionPlannerGoalConstPtr &goal){
	self.run();
}

}
