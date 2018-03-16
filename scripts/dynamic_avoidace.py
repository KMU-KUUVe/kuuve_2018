#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros

from std_msgs.msg import String
from obstacle_detector.msg import Obstacles

class Sequence(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1'])
        self.counter = 0

    def execute(self, userdata):
        rospy.loginfo('Executing state FOO')
        rospy.sleep(3000)
        return 'outcome1'

class DynamicAvoidance:
    def __init__(self):
        self.obstacles_date = Obstacles()
        self.pub = rospy.Publisher('write', String, queue_size=10)
        self.sub = rospy.Subscriber('raw_obstacles', Obstacles, self.obstacles_cb)

    def obstacles_cb(self, data):
        self.obstacles_data = data
        print(obstacles_data)

    def execute(self):
        rospy.init_node('dynamic_avoidance', anonymous=True)
        rate = rospy.Rate(100)
        while true:
            rate.sleep()

if __name__ == '__main__':
    try:
        dynamic_mission = DynamicAvoidance()
        dynamic_mission.execute()
    except rospy.ROSInterruptException:
        print(error)
        pass
