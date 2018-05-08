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

from action_with_smach.msg import MissionPlannerAction, MissionPlannerGoal, MissionPlannerResult, MissionPlannerFeedback
from dynamic_avoidance import DynamicAvoidance

def execute_cb(goal):
    rospy.loginfo("Goal Received")
    dynamic_mission = DynamicAvoidance()
    result = MissionPlannerResult()
    result.time_elapsed = rospy.Duration(1)
    dynamic_mission.execute()
    action_server.set_succeeded(result)
                
if __name__ == '__main__':
    rospy.init_node('dynamic_avoidance', anonymous=True)
    try:
        action_name = 'Mission1'
        action_server = actionlib.SimpleActionServer(action_name, MissionPlannerAction, execute_cb=execute_cb, auto_start=False)
        action_server.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        print(error)
        pass
