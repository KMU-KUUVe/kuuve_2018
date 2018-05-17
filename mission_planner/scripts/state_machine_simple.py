#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
from keyboard.msg import Key
import actionlib
from action_with_smach.msg import MissionPlannerAction, MissionPlannerGoal, MissionPlannerResult, MissionPlannerFeedback

mission_code = '\0'

#define state Foo
class Foo(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5'])
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
        while True:
            if self.key_value == 'q':
                return 'outcome3'
            elif self.key_value == '1':
                return 'outcome1'
            elif self.key_value == '2':
                return 'outcome2'
            elif self.key_value == '4':
                return 'outcome4'
            elif self.key_value == '5':
                return 'outcome5'

#define state Bar
class Bar(smach.State):
        def __init__(self, client_name):
                smach.State.__init__(self, outcomes=['outcome2'])
                self.client_name = client_name

        def execute(self, userdata):
                rospy.loginfo('Executing state BAR')
                client = actionlib.SimpleActionClient(self.client_name, MissionPlannerAction)
                client.wait_for_server()
                goal = MissionPlannerGoal() 
                goal.mission = 2
                rospy.loginfo("send goal")
                client.send_goal(goal)
                client.wait_for_result()
                rospy.loginfo('%s finish'%self.client_name)
                return 'outcome2'

#main
def main():
    rospy.init_node('smach_example_state_machine')

    #Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome6', 'outcome7'])

    #Open the container
    with sm:
        smach.StateMachine.add('MissionManager', Foo(), transitions={'outcome1':'dynamic_avoidance', 'outcome2':'u_turn', 'outcome3':'follow_line', 'outcome4':'narrow_path', 'outcome5':'static_avoidance'})
        smach.StateMachine.add('dynamic_avoidance', Bar('dynamic_avoidance'), transitions={'outcome2':'MissionManager'})
        smach.StateMachine.add('u_turn', Bar('u_turn'), transitions={'outcome2':'MissionManager'})
        smach.StateMachine.add('follow_line', Bar('follow_line'), transitions={'outcome2':'MissionManager'})
        smach.StateMachine.add('narrow_path', Bar('narrow_path'), transitions={'outcome2':'MissionManager'})
        smach.StateMachine.add('static_avoidance', Bar('static_avoidance'), transitions={'outcome2':'MissionManager'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    #Execute SMACH plan
    outcome = sm.execute()

    rospy.spin()
    sis.stop()
    

if __name__ == '__main__':
    main()
