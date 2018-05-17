#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
import actionlib 

from std_msgs.msg import String
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import SegmentObstacle 
from geometry_msgs.msg import Point

from mission_planner.msg import MissionPlannerAction, MissionPlannerGoal, MissionPlannerResult, MissionPlannerFeedback
from u_turn import Uturn 

def execute_cb(goal):
    rospy.loginfo("Goal Received")
    u_turn_mission = Uturn()
    result = MissionPlannerResult()
    result.time_elapsed = rospy.Duration(1)
    u_turn_mission.execute()
    action_server.set_succeeded(result)
		
if __name__ == '__main__':
    rospy.init_node('u_turn', anonymous=True)
    try:
        action_name = 'u_turn'
        action_server = actionlib.SimpleActionServer(action_name, MissionPlannerAction, execute_cb=execute_cb, auto_start=False)
        action_server.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        print(error)
        pass
