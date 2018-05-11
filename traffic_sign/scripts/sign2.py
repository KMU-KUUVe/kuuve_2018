#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


count = 0


img=np.zeros((300,512,3), np.uint8)

def nothing(x):
  pass


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_send",Image,queue_size = 20)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


   # if cols > 60 and rows > 60 :
   #   cv2.circle(cv_image, (50,50), 10, 255)

    #cv2.namedWindow('image') 

    #cv2. createTrackbar('R1','image',0,255,nothing)
    #cv2. createTrackbar('G1','image',0,255,nothing)
    #cv2. createTrackbar('B1','image',0,255,nothing)

    #cv2. createTrackbar('R2','image',0,255,nothing)
    #cv2. createTrackbar('G2','image',0,255,nothing)
    #cv2. createTrackbar('B2','image',0,255,nothing)

    #switch = '0 : OFF \n1 : ON'
    #cv2.createTrackbar(switch, 'image',0,1,nothing)

    #cv2.imshow('image',img)

    #imgray=cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    #ret2,thresh = cv2.threshold(imgray,127,255,0)
    
    #r1=cv2.getTrackbarPos('R1','image')
    #g1=cv2.getTrackbarPos('G1','image')
    #b1=cv2.getTrackbarPos('B1','image')
    #s1=cv2.getTrackbarPos(switch,'image')

    #r2=cv2.getTrackbarPos('R2','image')
    #g2=cv2.getTrackbarPos('G2','image')
    #b2=cv2.getTrackbarPos('B2','image')
    #s2=cv2.getTrackbarPos(switch,'image')


    (rows,cols,channels) = cv_image.shape

    #imCrop2 = cv_image[365:(365+160),430:(430+870)]

    imCrop2 = cv_image[cols/8:(cols/8+cols/3),rows/3:(rows/3+rows)]


    hsv=cv2.cvtColor(imCrop2, cv2.COLOR_BGR2HSV)
    lower_blue=np.array([105,50,50])	
    upper_blue=np.array([130,255,255])
    lower_red =np.array([0,70,50])
    upper_red =np.array([10,255,255])
    lower_red2 =np.array([170,70,50])
    upper_red2 =np.array([180,255,255])
    mask_b=cv2.inRange(hsv, lower_blue,upper_blue)
    mask_r1=cv2.inRange(hsv, lower_red,upper_red)
    mask_r2=cv2.inRange(hsv, lower_red2,upper_red2)

    mask_r = mask_r1 + mask_r2
    mask = mask_r + mask_b

    #res=cv2.bitwise_and(cv_image,cv_image,mask=mask)
    #imgray2=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #ret3,res_thresh= cv2.threshold(imgray2,127,255,0)


    #if s1 == 0:
     #   img[:] = 0
    #else:
     #   img[:] = [b1,g1,r1]

    #if s2 == 0:
     #   img[:] = 0
    #else:
     #   img[:] = [b2,g2,r2]



    image,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) > 0) :
        max_area = 0;
        ci = 0
        for i in range(len(contours)):
    	    cnt = contours[i]
    	    area = cv2.contourArea(cnt)
    	    if (area > max_area):
        	max_area = area
	        ci = i  
 
        cnt = contours[ci] 
    #cv2.drawContours(original_image, [contours[ci]], 0, (255, 0, 255), 3)

        epsilon = 0.1 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        x,y,w,h = cv2.boundingRect(approx)

        print('w: %d,h: %d' %(w,h))


        if w>25 and h>25 and w/h < 2 :


            cv2.rectangle(imCrop2,(x,y),(x+w,y+h),(0,255,0),2)

            imCrop = imCrop2[y:y+h,x:x+w]

	  # pub_image = cv2.resize(imCrop,(150,150))


            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(imCrop, "bgr8"))
                #self.image_pub.publish(self.bridge.cv2_to_imgmsg(pub_image, "bgr8"))

            except CvBridgeError as e:
                print(e)

            #cv2.imwrite(str,imCrop)

            cv2.imshow('result',imCrop)
 
        cv2.imshow('contour image',cv_image)
        cv2.imshow('roi',imCrop2)




    #cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)


def main(args):
  ic = image_converter()
  rospy.init_node('image_converter__', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
