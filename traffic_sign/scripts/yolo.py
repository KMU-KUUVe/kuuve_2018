#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
import tensorflow as tf
import sys
sys.path.insert(0, '/home/nvidia/Downloads/YOLO/nms')
sys.path.insert(0, '/home/nvidia/Downloads/YOLO/convet_method')

import file


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def yolo():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('chatter', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print('hello')
