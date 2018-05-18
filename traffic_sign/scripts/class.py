#!/usr/bin/env python
import sys
import rospy
import cv2
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Int32

imagePath = '/home/avees-server/sign.jpg'
modelFullPath = '/home/avees-server/catkin_ws/src/kuuve_2018/traffic_sign/output_graph.pb'
labelsFullPath ='/home/avees-server/catkin_ws/src/kuuve_2018/traffic_sign/output_labels.txt'

cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cnt5 = 0
cnt6 = 0
cnt7 = 0
cnt9 = 0

cntt = 2

pub = rospy.Publisher('sign', Int32, queue_size=10)

crosswalk_code = 1
u_turn_code = 2
static_avoidance_code = 3
dynamic_avoidance_code = 4
narrow_path_code = 5
s_path_code = 6
parking_code = 7

with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    sess = tf.Session() 
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def run_inference_on_image():
    global cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7, cnt9
    global cntt
    '''tt = 0
    if tt == 0:

        tt = 1
'''
  #  global pub
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
   
    predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        if score >= 0.7 :
            if human_string == "parking" :
                #rospy.loginfo("parking")
                cnt1 = cnt1 + 1 
            elif human_string == "narrow" :
                #rospy.loginfo("narrow")
                cnt2 = cnt2 + 1 
            elif human_string == "curve" :
                #rospy.loginfo("curve")
                cnt3 = cnt3 + 1 
            elif human_string == "static" :
                #rospy.loginfo("static_avoidance")
                cnt4 = cnt4 + 1 
            elif human_string == "dynamic" :
                #rospy.loginfo("dynamic_avoidance")
                cnt5 = cnt5 + 1 
            elif human_string == "corn" :
                #rospy.loginfo("corn")
                cnt6 = cnt6 + 1
            elif human_string == "uturn" :
                #rospy.loginfo("u_turn")
                cnt7 = cnt7 + 1 
            elif human_string == "pede" :
                #rospy.loginfo("pede")
                cnt9 = cnt9 + 1 
                #print('%s (score = %.5f)' % (human_string, score))
			    
    print("--------------------")

    print(cnt1,cnt2,cnt3,cnt4,cnt5,cnt6,cnt7,cnt9)
    if(cnt1>cntt) :
        #parking
        mode = 1
        rospy.loginfo("parking")
        pub.publish(parking_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")
        
    elif(cnt2>cntt) :
        mode = 2
        rospy.loginfo("narrow")
        pub.publish(narrow_path_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt3>cntt) :
        mode = 3
        rospy.loginfo("s_path")
        pub.publish(s_path_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt4>cntt) :
        mode = 4
        rospy.loginfo("static_avoidance")
        pub.publish(static_avoidance_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt5>cntt) :
        mode = 5
        rospy.loginfo("dynamic_avoidance")
        pub.publish(dynamic_avoidance_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt6>cntt * 2) :
        rospy.loginfo("corn")
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt7>cntt) :
        mode = 7
        rospy.loginfo("u_turn--")
        pub.publish(u_turn_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

    elif(cnt9>cntt) :
        mode = 9
        rospy.loginfo("crosswalk")
        pub.publish(crosswalk_code)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
	cnt6 = 0
        cnt7 = 0
        cnt9 = 0
        print("")

class classifier:
    def __init__(self):
        self.ccc = 0
        self.bridge=CvBridge()
        self.image_sub= rospy.Subscriber("test",Image,self.callback)



        
    def callback(self,data):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBrdigeError as e:
            print(e)
        
    #print("save_done")
        self.ccc +=1
        if self.ccc == 5:
            cv2.imwrite(imagePath, cv_image)
            run_inference_on_image()
            self.ccc = 0

def main(args):
    heybro= classifier()
    rospy.init_node('classifier', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shut down")
        sess.close()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    main(sys.argv)
