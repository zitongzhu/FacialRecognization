#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:45:13 2019

@author: zhuzitong
"""

import os
import sys
import numpy as np
import cv2
 
IMAGE_SIZE = 64
 
#Resize the images
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    #get the size
    h, w, _ = image.shape   
    #find longest edge
    longest_edge = max(h, w)        
    #calculate the length needed for width
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass   
    #Set RGB
    BLACK = [0, 0, 0]   
    #add length to width to make it to a square
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    return cv2.resize(constant, (height, width))
 
#Read trainning
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        
        #Get full path
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):    #if it is a folder
            read_path(full_path)
        else:   #image
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                if not image is None:             
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    
                    images.append(image)                
                    labels.append(path_name)                                
                    
    return images,labels
    
 
#Training
def load_dataset(path_name,username):    
    images,labels = read_path(path_name)    
    #get an array of (number of picture, length, width, number of color)
    images = np.array(images)
    print(images.shape)  
    
    #if it is my face, label 1
    labels = np.array([0 if label.endswith(username) else 1 for label in labels])    
    
    return images, labels
 
def main(name):
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, labels = load_dataset("/Users/zhuzitong/Desktop/iFace/image/train_faces",name) 


