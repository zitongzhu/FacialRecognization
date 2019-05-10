#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:32:29 2019

@author: zhuzitong
"""

import cv2
import sys
import os


def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)
            
def CatchPICFromVideo(window_name, camera_idx, catch_pic_num,username):
    path_name='/Users/zhuzitong/Desktop/iFace/image/train_faces/'+ username
    #Create path to save
    createdir(path_name)
    #Create window
    cv2.namedWindow(window_name)
    #Get video
    cap = cv2.VideoCapture(camera_idx)                 
    #Call facial detection from OpenCV
    classfier = cv2.CascadeClassifier("/Users/zhuzitong/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml") 
    #Draw a box for face
    color = (0, 255, 0)
    #Count number
    num = 0    
    #Detect face
    while cap.isOpened():
        #ok=T or F, frame is the pixels
        ok, frame = cap.read()
        if not ok:            
            break                
        #Trans the picture to grey
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)             
        
        #minNeighbor: valid points
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #if > 0, it is face                                
            for faceRect in faceRects:  #Frame the face
                #faceRect is a point with four parameter
                x, y, w, h = faceRect                        
                
                #Save the frame as a picture
                img_name = '%s/%d.jpg'%(path_name, num)      
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):
                    break
                
                #Draw the frame
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                #Show the number of face
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        if num > (catch_pic_num): break                
                       
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    cap.release()
    cv2.destroyAllWindows() 
    
def main(name):     
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("Face detection", 0,50,name)
        

         
