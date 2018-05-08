#! /usr/bin/env python

import rospy

import actionlib

from action_with_smach.msg import MissionPlannerAction, MissionPlannerGoal, MissionPlannerResult, MissionPlannerFeedback

class MissionAction(object):
    # create messages that are used to publish feedback/result
    _feedback = MissionPlannerFeedback()
    _result = MissionPlannerResult()

    def __init__(self, name):
        print("class init")
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, MissionPlannerAction, self.do_mission_cb, auto_start = False)
        self._as.start()
      
    def do_mission_cb(self, goal):
        # helper variables
        print("server start")
        r = rospy.Rate(1)
        success = True
        
        # publish info to the console for the user
        rospy.loginfo('%s: Executing, creating fibonacci sequence of order %i' % (self._action_name, goal.mission))
        rospy.sleep(goal.mission)

        if success:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(rospy.Duration())
        
if __name__ == '__main__':
    print("server1 init")
    rospy.init_node('Mission1')
    server = MissionAction('Mission1')
    rospy.spin()
