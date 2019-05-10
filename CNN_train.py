#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:30:06 2019

@author: zhuzitong
"""

import random
 
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
 
from loadData import load_dataset, resize_image, IMAGE_SIZE
 
 
class Dataset:
    def __init__(self, path_name,name):
        self.username=name
        #Trainning set
        self.train_images = None
        self.train_labels = None
        
        #Proof set
        self.valid_images = None
        self.valid_labels = None
        
        #Test set
        self.test_images  = None            
        self.test_labels  = None
        
        #path
        self.path_name    = path_name
        
        #dimensions order
        self.input_shape = None
        
    #The data sets were loaded and divided according to the principle of cross validation and preprocessed
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 3, nb_classes = 2):
        #Load the data set into memory
        images, labels = load_dataset(self.path_name,self.username)        
        
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))                
        
        #if the shape is'th'，then the order of putting images is：channels,rows,cols，otherwise :rows,cols,channels
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            
            
            #output
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            #Modle of loss function: categorical_crossentropy
            #Use nb_classes to trans to 2D
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            #Pixel data floating point for normalization
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            #put into [0,1]
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels

            
#CNN class            
class Model:
    def __init__(self,username):
        self.model = None
        self.MODEL_PATH = '/Users/zhuzitong/Desktop/iFace/model/%s.face.model.h5'%username
        
    #build model
    def build_model(self, dataset, nb_classes = 2):
        #build an empty model
        self.model = Sequential() 
        
        #add different layers into CNN
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', 
                                     input_shape = dataset.input_shape))    #1 Convolution2D
        self.model.add(Activation('relu'))                                  #2 activation
        
        self.model.add(Convolution2D(32, 3, 3))                             #3 Convolution2D                            
        self.model.add(Activation('relu'))                                  #4 activation
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5 Pooling
        self.model.add(Dropout(0.25))                                       #6 Dropout
 
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7  Convolution2D
        self.model.add(Activation('relu'))                                  #8  activation
        
        self.model.add(Convolution2D(64, 3, 3))                             #9  Convolution2D
        self.model.add(Activation('relu'))                                  #10 activation
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 Pooling
        self.model.add(Dropout(0.25))                                       #12 Dropout
 
        self.model.add(Flatten())                                           #13 Flatten
        self.model.add(Dense(512))                                          #14 Dense
        self.model.add(Activation('relu'))                                  #15 activation   
        self.model.add(Dropout(0.5))                                        #16 Dropout
        self.model.add(Dense(nb_classes))                                   #17 Dense
        self.model.add(Activation('softmax'))                               #18 softmax
        
        #output
        self.model.summary()
        
    #Train the model
    def train(self, dataset, batch_size = 20, nb_epoch = 10, data_augmentation = True):        
        sgd = SGD(lr = 0.01, decay = 1e-6, 
                  momentum = 0.9, nesterov = True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])   #compile model
        
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        #Change the dataset to improve
        else:            
            datagen = ImageDataGenerator(
                featurewise_center = False,             
                samplewise_center  = False,             
                featurewise_std_normalization = False,  
                samplewise_std_normalization  = False,  
                zca_whitening = False,                  
                rotation_range = 20,                    #Rotation angle
                width_shift_range  = 0.2,               #width shift
                height_shift_range = 0.2,               #height_shift
                horizontal_flip = True,                 #horizon flip
                vertical_flip = False)                  #no vertical flip

            datagen.fit(dataset.train_images)                        
 
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))    
    
    
    def save_model(self):
        file_path = self.MODEL_PATH
        self.model.save(file_path)
 
    def load_model(self):
        file_path = self.MODEL_PATH
        self.model = load_model(file_path)
 
    def evaluate(self, dataset):
         score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
         print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
          #Trainning set
         self.train_images = None
         self.train_labels = None
        
        #Proof set
         self.valid_images = None
         self.valid_labels = None
        
        #Test set
         self.test_images  = None            
         self.test_labels  = None
 

    def face_predict(self, image):    
        #Check the order of parameters
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                             #change it size to the same as train_set
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))       
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    
        
        image = image.astype('float32')
        image /= 255
        
        #ouput predict value
        result = self.model.predict_proba(image)
        print('result:', result)
        result = self.model.predict_classes(image)        
        return result[0]
 
 

def main1(username):
    dataset = Dataset('/Users/zhuzitong/Desktop/iFace/image/train_faces/',username)    
    dataset.load()
    model = Model(username)
    model.build_model(dataset)  
    model.build_model(dataset)
    model.train(dataset)

    
def main2(username):
    dataset = Dataset('/Users/zhuzitong/Desktop/iFace/image/train_faces/',username)    
    dataset.load()    
    model = Model(username)
    model.build_model(dataset)
    model.train(dataset)
    model.save_model()
   
def main3(username): 
    dataset = Dataset('/Users/zhuzitong/Desktop/iFace/image/train_faces/',username)    
    dataset.load()
    model = Model(username)
    model.load_model()
    model.evaluate(dataset)  

if __name__ == "__main__":
    name=input("Enter the name you want to train: ")
    main1(name)
    main2(name)
    main3(name)

