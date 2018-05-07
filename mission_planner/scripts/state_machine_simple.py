#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
from keyboard.msg import Key

mission_code = '\0'

"""
class KeyListener():
    def __init__(self):
        self.key_sub = rospy.Subscriber('keyboard/keydown', Key, self.keyboard_cb)

    def keyboard_cb(self, data):
        mission_code = char(data.code)
        print(mission_code)
        return mission_code
"""

#define state Foo
class Foo(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1', 'outcome2', 'outcome3'])
        self.key_value = '\0'

        self.key_sub = rospy.Subscriber('keyboard/keydown', Key, self.keyboard_cb, queue_size=1)

    def keyboard_cb(self, data):
        self.key_value = chr(data.code)
        print(self.key_value)

    def execute(self, userdata):
        rospy.loginfo('Executing state FOO')
        self.key_value = '\0'
        print("key value=")
        print(self.key_value)
        while 1:
            if self.key_value == 'q':
                return 'outcome3'
            elif self.key_value == '1':
                return 'outcome1'
            elif self.key_value == '2':
                return 'outcome2'

#define state Bar
class Bar(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome2'])

    def execute(self, userdata):
        rospy.loginfo('Executing state BAR')
        rospy.sleep(2)
        return 'outcome2'

#main
def main():
    rospy.init_node('smach_example_state_machine')

    #Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome4', 'outcome5'])

    #Open the container
    with sm:
        #smach.StateMachine.add('FOO', smach_ros.MonitorState("/keyboard/keydown", Key, keyboard_cb), transitions={'1':'BAR'})
        #smach.StateMachine.add('BAR', smach_ros.MonitorState("/keyboard/keydown", Key, keyboard_cb), transitions={'2':'FOO'})
        #Add states to the container
        smach.StateMachine.add('MissionManager', Foo(), transitions={'outcome1':'Mission1', 'outcome2':'Mission2', 'outcome3':'outcome4'})
        smach.StateMachine.add('Mission1', Bar(), transitions={'outcome2':'MissionManager'})
        smach.StateMachine.add('Mission2', Bar(), transitions={'outcome2':'MissionManager'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    #Execute SMACH plan
    outcome = sm.execute()

    rospy.spin()
    sis.stop()
    

if __name__ == '__main__':
    main()

