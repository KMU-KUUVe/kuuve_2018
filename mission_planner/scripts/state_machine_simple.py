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

crosswalk_stop_code = '1'
u_turn_code = '2'
static_avoidance_code = '3'
dynamic_avoidance_code = '4'
narrow_path_code = '5'
s_path_code = '6'
kuuve_parking_code = '7'

#define state MissionManager
class MissionManager(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['crosswalk_stop', 'u_turn', 'static_avoidance', 'dynamic_avoidance', 'narrow_path', 's_path', 'kuuve_parking'])
        self.key_value = mission_init_code 
        self.key_sub = rospy.Subscriber('keyboard/keydown', Key, self.keyboard_cb, queue_size=1)
        self.int_sub = rospy.Subscriber('sign', Int32, self.sign_cb, queue_size=10)
        self.goal = MissionPlannerGoal()
        self.client = actionlib.SimpleActionClient('lane_detector', MissionPlannerAction)
        self.nitro = False 

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
#self.client.wait_for_server()
        while not rospy.is_shutdown():
            self.goal.mission = 0
            key_str = ""
#rospy.loginfo('lane_detector goal = %d', self.goal.mission)
            if self.key_value == crosswalk_stop_code:
                key_str = 'crosswalk_stop'
            elif self.key_value == u_turn_code:
                key_str = 'u_turn'
            elif self.key_value == static_avoidance_code:
                rospy.sleep(1.0)
                key_str = 'static_avoidance'
            elif self.key_value == dynamic_avoidance_code:
                key_str = 'dynamic_avoidance'
            elif self.key_value == narrow_path_code:
                key_str = 'narrow_path'
            elif self.key_value == s_path_code:
                key_str = 'narrow_path'
            elif self.key_value == kuuve_parking_code:
                self.nitro = True
                rospy.loginfo('!!Nitro activated!!')
                key_str = 'kuuve_parking'
            else:
                if self.nitro:
                    self.goal.mission = 2;
                else:
                    self.goal.mission = 1;

            #send a goal that stop, go, nitro to lane_detector.
            self.client.send_goal(self.goal)

            if not key_str == "":
                return key_str
            r.sleep()

#define state Mission
class Missions(smach.State):
        def __init__(self, client_name):
                smach.State.__init__(self, outcomes=['finish'])
                self.client_name = client_name
                self.goal = MissionPlannerGoal() 

        def execute(self, userdata):
                rospy.loginfo('Executing state %s', self.client_name)
                client = actionlib.SimpleActionClient(self.client_name, MissionPlannerAction)
                client.wait_for_server()
                self.goal.mission = 2
                rospy.loginfo("send goal")
                client.send_goal(self.goal)
                client.wait_for_result()
                rospy.loginfo('%s finish'%self.client_name)
                return 'finish'

#main
def main():
    rospy.init_node('smach_example_state_machine')

    sm = smach.StateMachine(outcomes=[])

    #Open the container
    with sm:
        smach.StateMachine.add('MissionManager', MissionManager(), transitions={'crosswalk_stop':'crosswalk_stop', 'static_avoidance':'static_avoidance', 'narrow_path':'narrow_path', 's_path':'s_path', 'dynamic_avoidance':'dynamic_avoidance', 'u_turn':'u_turn','kuuve_parking':'kuuve_parking'})
        smach.StateMachine.add('crosswalk_stop', Missions('crosswalk_stop'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('u_turn', Missions('u_turn'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('dynamic_avoidance', Missions('dynamic_avoidance'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('static_avoidance', Missions('static_avoidance'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('narrow_path', Missions('narrow_path'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('s_path', Missions('s_path'), transitions={'finish':'MissionManager'})
        smach.StateMachine.add('kuuve_parking', Missions('kuuve_parking'), transitions={'finish':'MissionManager'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    #Execute SMACH plan
    outcome = sm.execute()

    rospy.spin()
    sis.stop()
    

if __name__ == '__main__':
    main()
