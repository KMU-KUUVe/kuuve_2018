#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
import math

from std_msgs.msg import String
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import CircleObstacle 
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped


class Uturn:
    def __init__(self):
        self.obstacles_date = Obstacles()
        self.pub = rospy.Publisher('ackermann', AckermannDriveStamped, queue_size=10)
        self.sub = rospy.Subscriber('raw_obstacles', Obstacles, self.obstacles_cb)
		self.obstacle_counter = 0

    def calc_distance(self, point):
        distance = math.sqrt((point.x)**2 + (point.y)**2)
        return distance

    def obstacles_cb(self, data):
        self.obstacles_data = data
		self.obstacle_counter = 0
        for obstacle in self.obstacles_data.circles:
		    if obstacle.center.x < 4.3:
			    self.obstacle_counter += 1
			    

    def execute(self):
        rospy.init_node('u_turn', anonymous=True)
        rate = rospy.Rate(100)
        acker_data = AckermannDriveStamped()

		acker_data.drive.steering_angle = -2
		acker_data.drive.speed = 0
		self.pub.publish(acker_data)
		rospy.sleep(1)

        while self.obstacle_counter < 5:
            print("approaching cones")
            acker_data.drive.steering_angle = -2
            acker_data.drive.speed = 7
            self.pub.publish(acker_data)
        print("too close")

        print("first left turn")
        acker_data.drive.steering_angle = -26
        acker_data.drive.speed = 6
        self.pub.publish(acker_data)
        rospy.sleep(13)

        print("right turn")
        acker_data.drive.steering_angle = 3
        acker_data.drive.speed = 6
        self.pub.publish(acker_data)
        rospy.sleep(1)

		print("finish")
        acker_data.drive.steering_angle = -2
        acker_data.drive.speed = 0
        self.pub.publish(acker_data)
        rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        u_turn_mission = Uturn()
        u_turn_mission.execute()
    except rospy.ROSInterruptException:
        print(error)
        pass
