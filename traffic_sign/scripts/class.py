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

imagePath = '/home/hdh7485/sign.jpg'
modelFullPath = '/home/hdh7485/catkin_ws/src/kuuve_2018/traffic_sign/output_graph.pb'
labelsFullPath ='/home/hdh7485/catkin_ws/src/kuuve_2018/traffic_sign/output_labels.txt'


pub = rospy.Publisher('sign', Int32, queue_size=10)

with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    sess = tf.Session() 
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def run_inference_on_image():
    tt = 0
    if tt == 0:
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt7 = 0
    cnt9 = 0
    mode = 0
    count = 0
    prev_mode =0
    tt = 1

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
    if score >= 0.3 :
        if human_string == "parking" :
            cnt1 = cnt1 + 1 
        elif human_string == "narrow" :
            print("narrow")
            pub.publish(mode)
            cnt2 = cnt2 + 1 
        elif human_string == "curve" :
            cnt3 = cnt3 + 1 
        elif human_string == "static" :
            cnt4 = cnt4 + 1 
        elif human_string == "dynamic" :
            cnt5 = cnt5 + 1 
        elif human_string == "uturn" :
            cnt7 = cnt7 + 1 
        elif human_string == "pede" :
            cnt9 = cnt9 + 1 
        print('%s (score = %.5f)' % (human_string, score))
    print("--------------------")


    if(cnt1>2) :
        mode = 1
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")
        
    elif(cnt2>2) :
        mode = 2
        rospy.info("narrow")
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")

    elif(cnt3>2) :
        mode = 3
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")
    elif(cnt4>2) :
        mode = 4
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")
    elif(cnt5>2) :
        mode = 5
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")
    elif(cnt7>2) :
        mode = 7
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")
    elif(cnt9>2) :
        mode = 9
        pub.publish(mode)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt7 = 0
        cnt9 = 0
        print(mode)
        print("")

class classifier:
    def __init__(self):
        self.ccc = 0
        self.bridge=CvBridge()
        #self.image_sub= rospy.Subscriber("test",Image,self.callback)
        self.image_sub= rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
    
        
    def callback(self,data):
    try:
        cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
    except CvBrdigeError as e:
        print(e)
        
    print("save_done")
    
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


