#!/usr/bin/env python

import roslib; roslib.load_manifest('smach_ros')
import rospy
import smach
import smach_ros
from keyboard.msg import Key
from std_msgs.msg import Int32
import actionlib
from mission_planner.msg import MissionPlannerAction, MissionPlannerGoal, MissionPlannerResult, MissionPlannerFeedback

mission_init_code = '\0'

crosswalk_code = '1'
u_turn_code = '2'
static_avoidance_code = '3'
dynamic_avoidance_code = '4'
narrow_path_code = '5'
s_path_code = '6'
parkinig_code = '7'

#define state MissionManager
class MissionManager(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['crosswalk', 'u_turn', 'static_avoidance', 'dynamic_avoidance', 'narrow_path', 's_path', 'parking'])
        self.key_value = mission_init_code 
        self.key_sub = rospy.Subscriber('keyboard/keydown', Key, self.keyboard_cb, queue_size=1)
        self.int_sub = rospy.Subscriber('sign', Int32, self.sign_cb, queue_size=10)

    def keyboard_cb(self, data):
        self.key_value = chr(data.code)
        rospy.loginfo(self.key_value)

    def sign_cb(self, int_msg):
        int_msg.data = int_msg.data + 48
        self.key_value = chr(int_msg.data)
        rospy.loginfo(self.key_value)
		
    def execute(self, userdata):
        rospy.loginfo('Executing state MissionManager')
        self.key_value = mission_init_code 
        rospy.loginfo("key value = %s", self.key_value)
        r = rospy.Rate(1000)
        while not rospy.is_shutdown():
            if self.key_value == crosswalk_code:
                return 'crosswalk'
            elif self.key_value == u_turn_code:
                return 'u_turn'
            elif self.key_value == static_avoidance_code:
                return 'static_avoidance'
            elif self.key_value == dynamic_avoidance_code:
                return 'dynamic_avoidance'
            elif self.key_value == narrow_path_code:
                return 'narrow_path'
            elif self.key_value == s_path_code:
                return 's_path'
            elif self.key_value == parking_code:
                return 'parking'
            r.sleep()

#define state Mission
class Missions(smach.State):
        def __init__(self, client_name):
                smach.State.__init__(self, outcomes=['finish'])
                self.client_name = client_name

        def execute(self, userdata):
                rospy.loginfo('Executing state %s', self.client_name)
                client = actionlib.SimpleActionClient(self.client_name, MissionPlannerAction)
                client.wait_for_server()
                goal = MissionPlannerGoal() 
                goal.mission = 2
                rospy.loginfo("send goal")
                client.send_goal(goal)
                client.wait_for_result()
                rospy.loginfo('%s finish'%self.client_name)
                return 'finish'

#main
def main():
    rospy.init_node('smach_example_state_machine')

    #Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome4', 'outcome5'])

    #Open the container
    with sm:
        smach.StateMachine.add('MissionManager', MissionManager(), transitions={'crosswalk':'crosswalk', 'static_avoidance':'static_avoidance', 'narrow_path':'narrow_path', 's_path':'s_path', 'dynamic_avoidance':'dynamic_avoidance', 'u_turn':'u_turn'})
        smach.StateMachine.add('crosswalk', Missions('crosswalk'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('u_turn', Missions('u_turn'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('dynamic_avoidance', Missions('dynamic_avoidance'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('static_avoidance', Missions('static_avoidance'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('narrow_path', Missions('narrow_path'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('s_path', Missions('s_path'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('parking', Missions('parking'), transitions={'finish':'MissionManager'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    #Execute SMACH plan
    outcome = sm.execute()

    rospy.spin()
    sis.stop()
    

if __name__ == '__main__':
    main()
