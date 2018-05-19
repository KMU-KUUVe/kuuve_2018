#!/usr/bin/env python
from __future__ import print_function
import roslib
#roslib.load_manifest('sign')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



def nothing(x):
  pass


class image_converter:

  def __init__(self):
    self.count = 0
    self.image_pub = rospy.Publisher("test",Image,queue_size = 20)
    self.img = np.zeros((10,400,3), np.uint8)
    self.box = np.zeros((200,400,3), np.uint8)
    
    
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)




  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      (rows,cols,channels) = cv_image.shape
      if self.count == 0 :
	cv2.namedWindow('image')
	cv2.namedWindow('box_size')


	cv2.createTrackbar('RHL','image',0,255,nothing)
	cv2.createTrackbar('RHH','image',0,255,nothing)
	cv2.setTrackbarPos('RHL','image',0)
	cv2.setTrackbarPos('RHH','image',90)

	cv2.createTrackbar('RSL','image',0,255,nothing)
	cv2.createTrackbar('RSH','image',0,255,nothing)
	cv2.setTrackbarPos('RSL','image',100)
	cv2.setTrackbarPos('RSH','image',255)

	cv2.createTrackbar('RVL','image',0,255,nothing)
	cv2.createTrackbar('RVH','image',0,255,nothing)
	cv2.setTrackbarPos('RVL','image',100)
	cv2.setTrackbarPos('RVH','image',255)

	cv2.createTrackbar('BHL','image',0,255,nothing)
	cv2.createTrackbar('BHH','image',0,255,nothing)
	cv2.setTrackbarPos('BHL','image',0)
	cv2.setTrackbarPos('BHH','image',0)

	cv2.createTrackbar('BSL','image',0,255,nothing)
	cv2.createTrackbar('BSH','image',0,255,nothing)
	cv2.setTrackbarPos('BSL','image',50)
	cv2.setTrackbarPos('BSH','image',255)

	cv2.createTrackbar('BVL','image',0,255,nothing)
	cv2.createTrackbar('BVH','image',0,255,nothing)
	cv2.setTrackbarPos('BVL','image',50)
	cv2.setTrackbarPos('BVH','image',255)

	cv2.createTrackbar('CHL','image',0,255,nothing)
	cv2.createTrackbar('CHH','image',0,255,nothing)
	cv2.setTrackbarPos('CHL','image',100)
	cv2.setTrackbarPos('CHH','image',130)

	cv2.createTrackbar('CSL','image',0,255,nothing)
	cv2.createTrackbar('CSH','image',0,255,nothing)
	cv2.setTrackbarPos('CSL','image',50)
	cv2.setTrackbarPos('CSH','image',255)

	cv2.createTrackbar('CVL','image',0,255,nothing)
	cv2.createTrackbar('CVH','image',0,255,nothing)
	cv2.setTrackbarPos('CVL','image',50)
	cv2.setTrackbarPos('CVH','image',255)

	cv2.createTrackbar('WL','box_size',0,255,nothing)
	cv2.createTrackbar('HL','box_size',0,255,nothing)
	cv2.setTrackbarPos('WL','box_size',50)
	cv2.setTrackbarPos('HL','box_size',50)

	cv2.createTrackbar('WH','box_size',0,255,nothing)
	cv2.createTrackbar('HH','box_size',0,255,nothing)
	cv2.setTrackbarPos('WH','box_size',200)
	cv2.setTrackbarPos('HH','box_size',200)

	'''
    lower_blue=np.array([105,50,50])	
    upper_blue=np.array([130,255,255])

    lower_red =np.array([0,70,50])
    upper_red =np.array([90,255,255])

    lower_red2 =np.array([170,70,50])
    upper_red2 =np.array([180,255,255])
	'''
	self.count += 1
      #print("got it")
    except CvBridgeError as e:
      print(e)
    cv2.imshow('image',self.img)
    cv2.waitKey(3)

    cv2.imshow('box_size',self.box)
    cv2.waitKey(3)

    #imCrop2 = cv_image[365:(365+160),430:(430+870)]

    imCrop2 = cv_image[cols/4:(cols/4+cols/4),rows/3:(rows/3+rows)]


    hsv=cv2.cvtColor(imCrop2, cv2.COLOR_BGR2HSV)


    rhl = cv2.getTrackbarPos('RHL','image')
    rsl = cv2.getTrackbarPos('RSL','image')
    rvl = cv2.getTrackbarPos('RVL','image')

    rhh = cv2.getTrackbarPos('RHH','image')
    rsh = cv2.getTrackbarPos('RSH','image')
    rvh = cv2.getTrackbarPos('RVH','image')

    lower_red = np.array([0,100,100])
    upper_red = np.array([14,255,255])

    mask_r1=cv2.inRange(hsv, lower_red,upper_red)

    bhl = cv2.getTrackbarPos('BHL','image')
    bsl = cv2.getTrackbarPos('BSL','image')
    bvl = cv2.getTrackbarPos('BVL','image')

    bhh = cv2.getTrackbarPos('BHH','image')
    bsh = cv2.getTrackbarPos('BSH','image')
    bvh = cv2.getTrackbarPos('BVH','image')

    lower_blue = np.array([0,50,20])
    upper_blue = np.array([0,255,255])

    mask_b=cv2.inRange(hsv, lower_blue,upper_blue)

    chl = cv2.getTrackbarPos('CHL','image')
    csl = cv2.getTrackbarPos('CSL','image')
    cvl = cv2.getTrackbarPos('CVL','image')

    chh = cv2.getTrackbarPos('CHH','image')
    csh = cv2.getTrackbarPos('CSH','image')
    cvh = cv2.getTrackbarPos('CVH','image')

    wh = cv2.getTrackbarPos('WH','box_size')
    wl = cv2.getTrackbarPos('WL','box_size')

    hh = cv2.getTrackbarPos('HH','box_size')
    hl = cv2.getTrackbarPos('HL','box_size')


    lower_ced = np.array([100,50,50])
    upper_ced = np.array([130,255,255])

    mask_r2=cv2.inRange(hsv, lower_ced,upper_ced)

    mask_r = mask_r1 + mask_r2
    mask = mask_r + mask_b

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

        
	
	div_ = w/(h *1.0)

	
        if w>80 and h>80 and w< 150 and h < 150 and div_ < 1.8:
	    print('w: %d,h: %d' %(w,h))

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
 
        #cv2.imshow('contour image',cv_image)
        #cv2.imshow('roi',imCrop2)




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


