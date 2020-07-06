#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:01:31 2020

@author: aarav
"""

import cv2
import numpy as np 
'''
    capture background
'''
vid = cv2.VideoCapture(0)
# let the camera get ready for action

def getBackground():
    global vid
    ret, frame = vid.read() 
    while(ret == True):
      #  cv2.imshow('background', frame) 
        return frame


def cloaked(frame, background):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    hsv_lower=np.array([36,100,100])     
    hsv_higher=np.array([86,255,255])
    green_mask=cv2.inRange(hsv, hsv_lower, hsv_higher)
  #  cv2.imshow('green_mask', green_mask)    
    frame[np.where(green_mask == 255)] = background[np.where(green_mask == 255)]
    #cv2.imshow('k', frame) 
    return frame

    
'''
working of code
'''

bg_img=getBackground()

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('original', frame) 
    cloaked_img=cloaked(frame, bg_img)
    cv2.imshow('cloaked', cloaked_img) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
    
    
    
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 