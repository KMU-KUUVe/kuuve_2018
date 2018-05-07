#! /usr/bin/env python

from __future__ import print_function
import rospy

# Brings in the SimpleActionClient
import actionlib
import actionlib_tutorials.msg

from keyboard.msg import Key

def keyboard_cb(key):
    print(key)
    if key.code == ord('q'):
        print('cancel')
        client.cancel_all_goals()
"""
    elif key.code == ord('w'):
        print('cancel 2, start 1')
        client2.cancel_all_goals()
"""

def feedback_cb(feedback):
    print(feedback)

rospy.Subscriber("keyboard/keydown", Key, keyboard_cb)
client = actionlib.SimpleActionClient('fibonacci', actionlib_tutorials.msg.FibonacciAction)
client2 = actionlib.SimpleActionClient('fibonacci2', actionlib_tutorials.msg.FibonacciAction)


def fibonacci_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (FibonacciAction) to the constructor.
    print("client start")
    client.wait_for_server()
    goal = actionlib_tutorials.msg.FibonacciGoal(order=20)
    print("send")
    client.send_goal(goal, feedback_cb=feedback_cb)
    print("wait")
    client.wait_for_result()
    print("result")
    return client.get_result()  # A FibonacciResult

def fibonacci_client2():
    # Creates the SimpleActionClient, passing the type of the action
    # (FibonacciAction) to the constructor.
    print("client start")
    client2.wait_for_server()
    goal = actionlib_tutorials.msg.FibonacciGoal(order=10)
    print("send")
    client2.send_goal(goal, feedback_cb=feedback_cb)
    print("wait")
    client2.wait_for_result()
    print("result")
    return client2.get_result()  # A FibonacciResult

if __name__ == '__main__':
    try:
        rospy.init_node('fibonacci_client_py')
        result = fibonacci_client()
        print("Result:", ', '.join([str(n) for n in result.sequence]))
        result2 = fibonacci_client2()
        print("Result2:", ', '.join([str(n) for n in result2.sequence]))

    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)

    rospy.spin()
