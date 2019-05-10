#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:31:56 2019

@author: zhuzitong
"""


import cv2
import sys
from CNN_model import Model
import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

 
emotion_model=load_model('/Users/zhuzitong/Desktop/iFace/checkpoint/emotion.model.h5')

def emotion_analysis(list1,list2,emotion_type):
    if emotion_type == 'angry':
        index=0
    elif emotion_type == 'disgust':
        index=1
    elif emotion_type == 'fear':
        index=2
    elif emotion_type == 'happy':
        index=3
    elif emotion_type == 'sad':
        index=4
    elif emotion_type == 'surprise':
        index=5
    elif emotion_type == 'neutral':
        index=6
    else:
        return -1
    value_list1=get_average_emotion(list1)
    value_list2=get_average_emotion(list2)
    for i in range(0,7):
        if value_list1[index]>=value_list1[i] and value_list2[index]>=value_list2[i]:
            pass
        else:
            return 1
    return 2
    
def get_average_emotion(list1):
    sum_list1=[0,0,0,0,0,0,0]
    for i in list1:              
        x = image.img_to_array(i)
        x = np.expand_dims(x, axis = 0)
        x /= 255
        emotion_custom = emotion_model.predict(x)
        for j in range(0,7):
            sum_list1[j]=sum_list1[j]+emotion_custom[0][j]
    for k in range(0,7):
        if not len(list1)==0:
            sum_list1[k]=sum_list1[k]/len(list1)
    return sum_list1
        
def emotion_diagram(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects)) 
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')    
    plt.show()
    
def main(username1,username2,emotion):
    
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
       
    #load model
    path = "/Users/zhuzitong/Desktop/iFace/model/"
    files= os.listdir(path) 
    #s = []
    modellist=[]
    namelist=[]
    
    for file_path in files:
        namepath=file_path.split('.')
        name=namepath[0]
        if name==username1 or name==username2:
            model = Model(name)
            model.load_model()
            modellist.append(model)
            namelist.append(name)
        else:
            pass
    
    '''
    for file_path in files:
        print(file_path)
        if file_path=='.face.model.h5' or file_path=='.DS_Store':
            pass
        else:
            namepath=file_path.split('.')
            name=namepath[0]
            model = Model(name)
            model.load_model()
            modellist.append(model)
            namelist.append(name)
    '''
    index1=namelist.index(username1)
    index2=namelist.index(username2)
    #circle face      
    color = (0, 255, 0)
    
    #call camera
    cap = cv2.VideoCapture(0)
    
    #call haar
    cascade_path = "/Users/zhuzitong/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"    
    count=0
    count1=0
    count2=0
    emotion_list1=[]
    emotion_list2=[]
    print(modellist)
    #face recognition
    while True:
        
        ret, frame = cap.read()        
        if ret is True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #load haar
        cascade = cv2.CascadeClassifier(cascade_path)                
 
        #find face
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #input face to recognize
                image_ = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                
                for i in range(0,2):
                    faceID = modellist[i].face_predict(image_)                 
                    #if it is me
                    if faceID == 0: 
                        count=count+1                                                      
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)                 
                        #mark my name
                        cv2.putText(frame,namelist[i], (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2)
                        if i == index1:
                            try:
                                img=cv2.resize(image_, (48, 48))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                                emotion_list1.append(img)
                                count1+=1
                            except Exception as e:
                                print(str(e))
                        elif i==index2:
                            try:
                                img=cv2.resize(image_, (48, 48))
                                img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                                emotion_list2.append(img)
                                count2+=1
                            except Exception as e:
                                print(str(e))
                    else:
                        pass
                            
        cv2.imshow("Face Recognition", frame)
        result=0
        if count>50:
            print('emotion for person1 is :\n')
            print(get_average_emotion(emotion_list1))
            emotion_diagram(get_average_emotion(emotion_list1))
            print('emotion for person2 is :\n')
            print(get_average_emotion(emotion_list2))
            emotion_diagram(get_average_emotion(emotion_list2))
            if count1>15 and count2>15:
                result=emotion_analysis(emotion_list1,emotion_list2,emotion)
            break
        
        
            
        #if want to quit
        k = cv2.waitKey(30)
        if k & 0xFF == ord('q'):
            break
    
    print("The recognition result is: %s"%result)
    print(count1)
    print(count2)

    cap.release()
    cv2.destroyAllWindows()
    return result
    
#main('zitongzhu','dijin','happy')