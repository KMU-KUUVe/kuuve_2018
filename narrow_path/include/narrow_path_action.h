#include <narrow_path.h>
#include <actionlib/server/simple_action_server.h>
#include <mission_planner/MissionPlannerAction.h>

namespace narrow_path_action{

class NarrowPathAction:public narrow_path::NarrowPath{
	public:
		NarrowPathAction(std::string name);
		//NarrowPathAction(std::string name, ros::NodeHandle nh) : NarrowPath(nh);
		void action_cb(const mission_planner::MissionPlannerGoalConstPtr &goal);

	private:
		actionlib::SimpleActionServer<mission_planner::MissionPlannerAction> as_;
		std::string name_;
};
}
