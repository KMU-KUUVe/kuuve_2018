#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
import math

from std_msgs.msg import String
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import SegmentObstacle 
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped


class Uturn:
    def __init__(self):
        self.obstacles_date = Obstacles()
        self.pub = rospy.Publisher('ackermann', AckermannDriveStamped, queue_size=10)
        self.sub = rospy.Subscriber('raw_obstacles', Obstacles, self.obstacles_cb)
#        self.nearest_obstacle = self.obstacles_data.segments[0]
        self.nearest_obstacle = SegmentObstacle()
        self.nearest_center_point = Point(100, 0, 0)

    def calc_distance(self, point):
        distance = math.sqrt((point.x)**2 + (point.y)**2)
        return distance

    def obstacles_cb(self, data):
        self.nearest_obstacle = SegmentObstacle()
        self.nearest_center_point = Point(100, 0, 0)
        self.obstacles_data = data
        for obstacle in self.obstacles_data.circles:
            self.center_point = Point() 
            self.center_point = obstacle.center
            '''
            self.center_point.x = (obstacle.first_point.x + obstacle.last_point.x)/2 
            self.center_point.y = (obstacle.first_point.y + obstacle.last_point.y)/2 
'''
            if self.calc_distance(self.nearest_center_point) > self.calc_distance(self.center_point) and self.center_point.x > 0 and self.center_point.y > -0.5 and self.center_point.y < 0.5:
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
        acker_data = AckermannDriveStamped()
        while self.nearest_center_point.x > 5.0:
            acker_data.drive.steering_angle = 0
            acker_data.drive.speed = 7
            self.pub.publish(acker_data)
        print("too close")

        print("turn start")
        #self.pub.publish("1100,1530,")
        acker_data.drive.steering_angle = 26
        acker_data.drive.speed = 6
        self.pub.publish(acker_data)
        rospy.sleep(13)

        print("finish")
        acker_data.drive.steering_angle = 0
        acker_data.drive.speed = 0
        self.pub.publish(acker_data)
        rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        u_turn = Uturn()
        u_turn.execute()
    except rospy.ROSInterruptException:
        print(error)
        pass
