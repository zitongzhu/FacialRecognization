# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:37:59 2019

@author: Lenovo
"""

import tkinter
import getFace
import face_recognition
import tkinter.messagebox

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.canvas = tkinter.Canvas(width = 400, height = 300)        
        self.canvas.pack()
        self.createButton()
        self.window.mainloop()
    
        
    def capture(self):
        name=self.name.get()
        if not name=='':
            getFace.main(name)
        return
        
    def recog(self):
        name2=self.name2.get()
        name3=self.name3.get()
        emo=self.emotion.get()
        result=face_recognition.main(name2,name3,emo)
        if result==2:
            tkinter.messagebox.showinfo('Result','Unlocked!')
        else:
            tkinter.messagebox.showinfo('Result','Recognition failed:(')
        return
    '''
    def train(self):
        name=self.name.get()
        CNN_train.main1(name)
        CNN_train.main2(name)
        CNN_train.main3(name)
        return
    '''
    def createButton(self):
        self.input = tkinter.Label(text = "Your name")
        self.input.pack(anchor=tkinter.CENTER, expand=True)
        
        self.name = tkinter.Entry(bd = 5)
        self.name.pack(anchor=tkinter.CENTER, expand=True)

        self.cap = tkinter.Button(text="Capture", width=100, command=self.capture)
        self.cap.pack(anchor=tkinter.CENTER, expand=True)
        '''
        self.train=tkinter.Button(text="Train", width=100, command=self.train)
        self.train.pack(anchor=tkinter.CENTER, expand=True)
        '''
        self.input2 = tkinter.Label(text = "name1")
        self.input2.pack(anchor=tkinter.CENTER, expand=True)
        self.name2 = tkinter.Entry(bd = 5)
        self.name2.pack(anchor=tkinter.CENTER, expand=True)
        
        self.input3 = tkinter.Label(text = "name2")
        self.input3.pack(anchor=tkinter.CENTER, expand=True)
        self.name3 = tkinter.Entry(bd = 5)
        self.name3.pack(anchor=tkinter.CENTER, expand=True)
        
        self.input4 = tkinter.Label(text = "Emotion Required(angry/disgust/fear/happy/sad/surprise/neutral)")
        self.input4.pack(anchor=tkinter.CENTER, expand=True)
        self.emotion = tkinter.Entry(bd = 5)
        self.emotion.pack(anchor=tkinter.CENTER, expand=True)
        
        
        self.reco=tkinter.Button(text="Reognition", width=100, command=self.recog)
        self.reco.pack(anchor=tkinter.CENTER, expand=True)

            
Application(tkinter.Tk(), "Face recognition")