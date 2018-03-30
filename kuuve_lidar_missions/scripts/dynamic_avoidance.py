#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros

from std_msgs.msg import String
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import SegmentObstacle 
from geometry_msgs.msg import Point


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
#        self.nearest_obstacle = self.obstacles_data.segments[0]
        self.nearest_obstacle = SegmentObstacle()
        self.nearest_center_point = Point(100, 0, 0)

    def calc_distance(self, point):
        distance = (point.x)**2 + (point.y)**2
        return distance

    def obstacles_cb(self, data):
        self.nearest_obstacle = SegmentObstacle()
        self.nearest_center_point = Point(100, 0, 0)
        self.obstacles_data = data
        for obstacle in self.obstacles_data.segments:
            self.center_point = Point() 
            self.center_point.x = (obstacle.first_point.x + obstacle.last_point.x)/2 
            self.center_point.y = (obstacle.first_point.y + obstacle.last_point.y)/2 

            if self.calc_distance(self.nearest_center_point) > self.calc_distance(self.center_point) and self.center_point.x > 0 and self.center_point.y > -0.3 and self.center_point.y < 0.3:
                self.nearest_center_point = self.center_point
                self.nearest_obstacle = obstacle
                '''
        print(self.nearest_center_point)
        print(self.nearest_obstacle)
        print('-------------------')
'''

    def execute(self):
        rospy.init_node('dynamic_avoidance', anonymous=True)
        rate = rospy.Rate(100)
        while self.nearest_center_point.x > 0.4:
            self.pub.publish("1500,1520,")
            rate.sleep()
        print("too close")
        while self.nearest_center_point.x < 0.5:
            self.pub.publish("1500,1500,")
            rate.sleep()
        print("obstacle dissapear")
        self.pub.publish("1500,1528,")
        rospy.sleep(2)
        self.pub.publish("1500,1500,")

if __name__ == '__main__':
    try:
        dynamic_mission = DynamicAvoidance()
        dynamic_mission.execute()
    except rospy.ROSInterruptException:
        print(error)
        pass
