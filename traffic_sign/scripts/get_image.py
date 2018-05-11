#!/usr/bin/env python

import rospy
import sys
import numpy as np
import cv2
import argparse
#import matplotlib.pyplot as plt
def _float_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer([filenames])
    reader = tf.TFRecordReader()
    _,tfrecord_serialized = reader.read(tfrecord_file_queue)
    
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label':tf.VarLenFeature(dtype=tf.float32),
                                                    'image':tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    image = tf.decode_raw(tfrecord_features['image'], tf.float32)
    label = tf.sparse_tensor_to_dense(tfrecord_features['label'], default_value=0)
    

    image = tf.reshape(image, [416,416,3])
    label = tf.reshape(label, [13,13,5,25])
    image,label  = tf.train.shuffle_batch([image,label], batch_size=BATCH_SIZE,capacity=32, num_threads=8, min_after_dequeue=10)
    

    return image, label

def read_from_tfrecord_classification(filenames):
    tfrecord_file_queue = tf.train.string_input_producer([filenames])
    reader = tf.TFRecordReader()
    _,tfrecord_serialized = reader.read(tfrecord_file_queue)
    
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label':tf.VarLenFeature(dtype=tf.float32),
                                                    'image':tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    image = tf.decode_raw(tfrecord_features['image'], tf.float32)
    label = tf.sparse_tensor_to_dense(tfrecord_features['label'], default_value=0)
    image = tf.reshape(image, [416,416,3])
    label = tf.reshape(label, [20])
    image,label  = tf.train.shuffle_batch([image,label], batch_size=BATCH_SIZE,capacity=32, num_threads=8, min_after_dequeue=10)
    return image, label

def create_variable(shape):
    
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape),name='weight', trainable = False)
    return variable

def create_variable_(shape):
    
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape),name='weight')
    return variable

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def bias_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape),name='bias')
    return variable

def avgpool2d(x, size, strides):
    
    return tf.nn.avg_pool(x, ksize=[1,size,size,1], strides=[1,strides,strides,1], padding='SAME',data_format='NHWC')

def cal_logits(input_, W, b):
    return tf.matmul(input_, W) + b

def fc_layer(L, a, b, input_, W, bias):

    L_Flat = tf.reshape(L, [ -1,a * b * input_ ])

    logits = cal_logits(L_Flat, W, bias)
    return logits

def conv2d(input_, w, strides):
    
    
    output = tf.nn.conv2d(input_, w, strides=[1, strides, strides, 1], padding='SAME')

    return tf.nn.leaky_relu(output, alpha=0.1)

def maxpool2d(x, size, strides):
    return tf.nn.max_pool(x, ksize=[1,size,size,1], strides=[1,strides,strides,1], padding = 'SAME')


#import sys
#sys.path.append('/home/nvidia/Downloads/YOLO/nms')

#import nms
#sys.path.append('/home/nvidia/Downloads/YOLO/conv_method')
#import conv

GRID_W, GRID_H = 13, 13
ANCHORS =  1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071
BOX = 5
BATCH_SIZE = 1

windowName = "CameraDemo"
LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

from PIL import Image
import tensorflow as tf

tf.reset_default_graph()

def YOLOnet(images):

    with tf.variable_scope("shared_layer"):
        training = tf.placeholder_with_default(False, shape=(), name='training')
        w1 = create_variable([3,3,3,32])
        w2 = create_variable([3,3,32,64])


        w3 = create_variable([3,3,64,128])
        w4 = create_variable([1,1,128,64])
        w5 = create_variable([3,3,64,128])

        w6 = create_variable([3,3,128,256])
        w7 = create_variable([1,1,256,128])
        w8 = create_variable([3,3,128,256]) 

        w9 = create_variable([3,3,256,512])
        w10 = create_variable([1,1,512,256])
        w11 = create_variable([3,3,256,512])
        w12 = create_variable([1,1,512,256]) 
        w13 = create_variable([3,3,256,512])

        w14 = create_variable([3,3,512,1024])
        w15 = create_variable([1,1,1024,512])
        w16 = create_variable([3,3,512,1024])
        w17 = create_variable([1,1,1024,512])
        w18 = create_variable([3,3,512,1024])

    #    w19 = create_variable([1,1,1024,100])

        layer1 = conv2d(images, w1, 1)
        layer1 = tf.layers.batch_normalization(layer1, training = training, momentum=0.9)
        layer1_ = maxpool2d(layer1, 2, 2)

        layer2 = conv2d(layer1_, w2, 1)
        layer2 = tf.layers.batch_normalization(layer2, training = training, momentum=0.9)
        layer2_ = maxpool2d(layer2, 2, 2)

        layer3 = conv2d(layer2_, w3, 1)
        layer3 = tf.layers.batch_normalization(layer3, training = training, momentum=0.9)

        layer4 = conv2d(layer3, w4, 1)
        layer4 = tf.layers.batch_normalization(layer4, training = training, momentum=0.9)

        layer5 = conv2d(layer4, w5, 1)
        layer5 = tf.layers.batch_normalization(layer5, training = training, momentum=0.9)
        layer5_ = maxpool2d(layer5, 2, 2)

        layer6 = conv2d(layer5_, w6, 1)  
        layer6 = tf.layers.batch_normalization(layer6, training = training, momentum=0.9)

        layer7 = conv2d(layer6, w7, 1)
        layer7 = tf.layers.batch_normalization(layer7, training = training, momentum=0.9)

        layer8 = conv2d(layer7, w8, 1)
        layer8 = tf.layers.batch_normalization(layer8, training = training, momentum=0.9)
        layer8_ = maxpool2d(layer8, 2, 2)

        layer9 = conv2d(layer8_, w9, 1)
        layer9 = tf.layers.batch_normalization(layer9, training = training, momentum=0.9)

        layer10 = conv2d(layer9, w10, 1)
        layer10 = tf.layers.batch_normalization(layer10, training = training, momentum=0.9)

        layer11 = conv2d(layer10, w11, 1)
        layer11 = tf.layers.batch_normalization(layer11, training = training, momentum=0.9)

        layer12 = conv2d(layer11, w12, 1)
        layer12 = tf.layers.batch_normalization(layer12, training = training, momentum=0.9)

        layer13 = conv2d(layer12, w13, 1)
        layer13 = tf.layers.batch_normalization(layer13, training = training, momentum=0.9)
        layer13_ = maxpool2d(layer13, 2, 2)

        layer14 = conv2d(layer13_, w14, 1)
        layer14 = tf.layers.batch_normalization(layer14, training = training, momentum=0.9)

        layer15 = conv2d(layer14, w15, 1)
        layer15 = tf.layers.batch_normalization(layer15, training = training, momentum=0.9)

        layer16 = conv2d(layer15, w16, 1)
        layer16 = tf.layers.batch_normalization(layer16, training = training, momentum=0.9)

        layer17 = conv2d(layer16, w17, 1)
        layer17 = tf.layers.batch_normalization(layer17, training = training, momentum=0.9)

        layer18 = conv2d(layer17, w18, 1)
        layer18 = tf.layers.batch_normalization(layer18, training = training, momentum=0.9)

    with tf.variable_scope("detection_output_layer"):

        w19 = create_variable_([1,1,1024,1024])
        w20 = create_variable_([3,3,1024,1024])
        w21 = create_variable_([3,3,1024,1024])
        w22 = create_variable_([1,1,1024,125])


        layer19 = conv2d(layer18, w19, 1)
        layer19 = tf.layers.batch_normalization(layer19, training = training, momentum=0.9)

        layer20 = conv2d(layer19, w20, 1)
        layer20 = tf.layers.batch_normalization(layer20, training = training, momentum=0.9)

        layer21 = conv2d(layer20, w21, 1)
        layer21 = tf.layers.batch_normalization(layer21, training = training, momentum=0.9)

        layer22 = conv2d(layer21, w22, 1)
        layer22 = tf.layers.batch_normalization(layer22, training = training, momentum=0.9)

        pred = tf.reshape(layer22, [-1,13,13,5,25], name='pred')

	cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
	cell_y = tf.transpose(cell_x, (0,2,1,3,4))
	cell_grid = tf.tile(tf.concat([cell_y,cell_x], -1), [BATCH_SIZE, 1, 1, 5, 1])

	pred_box_xy = tf.sigmoid(pred[..., 0:2]) + cell_grid
	pred_box_wh = tf.exp(pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])/ 416.
	pred_box_wh_sqrt = tf.cast(tf.sqrt(pred_box_wh),tf.float32)
	pred_box_conf = tf.expand_dims(tf.sigmoid(pred[...,4]), -1)
	pred_box_prob = tf.nn.softmax(pred[...,5:])

	y_pred = tf.concat([pred_box_xy , pred_box_wh_sqrt, pred_box_conf], 4)
	inference_test = tf.concat([y_pred,pred_box_prob], 4)
	
    return inference_test

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2

    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h
    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union

def interval_overlap(interval_a, interval_b):

    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def decode_netout(netout, nms_threshold, obj_threshold, anchors, conf, nb_class):

    boxes = []

    for row in range(13):
        for col in range(13):
            for b in range(5):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                #if classes.any():
                if np.max(classes) > obj_threshold:
                    if netout[row,col,b,4] > conf:
                
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        #x = (col + sigmoid(x)) / GRID_W # center position, unit: image width
                        #y = (row + sigmoid(y)) / GRID_H # center position, unit: image height
                        x = x * 32. / 416.
                        y = y * 32. / 416.

                        w = w*w 
                        h = h*h 
                        #w = anchors[2 * b + 0] * np.exp(w) / GRID_W # unit: image width
                        #h = anchors[2 * b + 1] * np.exp(h) / GRID_H # unit: image height
                        confidence = netout[row,col,b,4]

                        box = BoundBox(x, y, w, h, confidence, classes)

                        boxes.append(box)
                    
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                                


    #boxes = [box for box in boxes if box.get_score() > obj_threshold and box.c > conf]
    #boxes = [box for box in boxes if box.c > conf]

                    
                    

    return boxes

def draw_boxes(image, boxes, labels):
    
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,255), 5)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2 *1e-3 * image.shape[0], 
                    (0,255,0), 2)
    
        
    return image 

def sigmoid(x):
    return 1. / (1. + np.exp(-x))




def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description=
                                     "Capture and display live camera video on Jetson TX2/TX1")
    parser.add_argument("--rtsp", dest="use_rtsp",
                        help="use IP CAM (remember to also set --uri)",
                        action="store_true")
    parser.add_argument("--uri", dest="rtsp_uri",
                        help="RTSP URI string, e.g. rtsp://192.168.1.64:554",
                        default=None, type=str)
    parser.add_argument("--latency", dest="rtsp_latency",
                        help="latency in ms for RTSP [200]",
                        default=200, type=int)
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam (remember to also set --vid)",
                        action="store_true")
    parser.add_argument("--vid", dest="video_dev",
                        help="video device # of USB webcam (/dev/video?) [1]",
                        default=1, type=int)
    parser.add_argument("--width", dest="image_width",
                        help="image width [1920]",
                        default=1920, type=int)
    parser.add_argument("--height", dest="image_height",
                        help="image width [1080]",
                        default=1080, type=int)
    args = parser.parse_args()
    return args


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("v4l2src device=/dev/video{} ! "
               "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
               "videoconvert ! appsink").format(dev, width, height)

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_FPS, 5)
    return cap 

def open_window(windowName, width, height):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, width, height)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, "Camera Demo for Jetson TX2/TX1")


def read_cam(windowName, cap):
    showHelp = True
    showFullScreen = False
    helpText = "'Esc' to Quit, 'H' to Toggle Help, 'F' to Toggle Fullscreen"
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
            # This will fail if the user closed the window; Nasties get printed to the console
            break;
        ret_val, displayBuf = cap.read();
        if showHelp == True:
            cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
            cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        
	displayBuf = cv2.resize(displayBuf, (416,416), interpolation=cv2.INTER_AREA)
	displayBuf_ = np.reshape(displayBuf, [-1,416,416,3])

	
	pred_ = sess.run(pred, feed_dict ={images: displayBuf_})

	#print(np.shape(pred_))
	print("test")
    	displayBuf_ = displayBuf_.astype(dtype=np.uint8)
    	boxes = decode_netout(pred_[0], 0.5,0.8 ,ANCHORS,0.2, 20)
   	print(np.shape(boxes))
    	for box in boxes:
            print(box.c, LABELS[np.argmax(box.classes)])
	    print(box.x, box.y, box.w, box.h)
	print(np.shape(displayBuf_))
    	image = draw_boxes(displayBuf_[0], boxes, LABELS)
 	#print(np.shape(image))



	cv2.imshow(windowName, image)

	

        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
	    print('close')
	    sess.close()
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            showHelp = not showHelp
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            showFullScreen = not showFullScreen
            if showFullScreen == True: 
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) 



if __name__ == '__main__':
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1,416,416,3], name="images")
    pred = YOLOnet(images)
    init_op = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, '/home/nvidia/Downloads/weight/train-90-16027')
    try:

        args = parse_args()
        print("Called with args:")
        print(args)
        print("OpenCV version: {}".format(cv2.__version__))
	cap = open_cam_usb(args.video_dev, args.image_width, args.image_height)



	if not cap.isOpened():
            sys.exit("Failed to open camera!")
	#open_window(windowName, args.image_width, args.image_height)
	open_window(windowName, 416, 416)
        read_cam(windowName, cap)
    

    except rospy.ROSInterruptException:
	print('test')
        cap.release()
        cv2.destroyAllWindows()
	sess.close()


