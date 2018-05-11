#!/usr/bin/env python

import rospy
import cv2
import tensorflow as tf
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        tensor_ready()

	get_raw_image_and_send()

	yolo_process_and_show_result()	
		
    except rospy.ROSInterruptException:
        pass
