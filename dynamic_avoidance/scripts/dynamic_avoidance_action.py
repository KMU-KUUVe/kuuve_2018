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
'''
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
        debug = 0
        if debug:
            print(self.nearest_center_point)
            print(self.nearest_obstacle)
            print('-------------------')

    def execute(self):
        rospy.loginfo("Mission Start!")
        rate = rospy.Rate(100)
        while self.nearest_center_point.x > 0.4:
            self.pub.publish("1500,1520,")
        rospy.loginfo("too close")
        while self.nearest_center_point.x < 0.5:
            self.pub.publish("1500,1500,")
        rospy.loginfo("obstacle dissapear")
        self.pub.publish("1500,1528,")
        rospy.sleep(2)
        self.pub.publish("1500,1500,")
'''

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
