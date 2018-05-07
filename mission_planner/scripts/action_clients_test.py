#! /usr/bin/env python

from __future__ import print_function
import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.
import actionlib_tutorials.msg

def feedback_cb(feedback):
    print(feedback)

def fibonacci_client():
    client = actionlib.SimpleActionClient('fibonacci', actionlib_tutorials.msg.FibonacciAction)

    client.wait_for_server()
    goal = actionlib_tutorials.msg.FibonacciGoal(order=20)
    client.send_goal(goal, feedback_cb=feedback_cb)
    client.wait_for_result()

    return client.get_result()  # A FibonacciResult

def fibonacci_client2():
    client2 = actionlib.SimpleActionClient('fibonacci2', actionlib_tutorials.msg.FibonacciAction)

    client2.wait_for_server()
    goal = actionlib_tutorials.msg.FibonacciGoal(order=10)
    client2.send_goal(goal, feedback_cb=feedback_cb)
    client2.wait_for_result()

    return client2.get_result()  # A FibonacciResult

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        print("Initialize client node")
        rospy.init_node('fibonacci_client_py')

        print("Wait for server1")
        result = fibonacci_client()
        print("Result:", ', '.join([str(n) for n in result.sequence]))

        print("Wait for server2")
        result2 = fibonacci_client2()
        print("Result2:", ', '.join([str(n) for n in result2.sequence]))

    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
