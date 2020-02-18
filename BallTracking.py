#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:44:26 2020

@author: Jiale Hu
"""

import time
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def BallTracking(vidDir, show=False):

    # Initialize ball position array
    position = np.empty((0,2));
    
    # Image generation parameters
    imageStep = 20
    imageList = []
    
    # Image filter parameters (HSV range)
    lowerB = (0,50,128)
    upperB = (20,140,248)
    kernel = np.ones((6, 6), np.uint8)
    
    # Start loading video
    start = time.time()
    
    cap = cv2.VideoCapture(vidDir)
    numFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cntFrame = 0
    
    # ret, frame = cap.read()
    # frame = cv2.blur(frame, (20, 20))
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # thresh = cv2.inRange(hsv, lowerB , upperB)
    # cv2.imshow('thresh', thresh)
    
    while True:
        
        cntFrame += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame resolution
        H, W, channels = frame.shape
        
        # Filtering
        blur = cv2.blur(frame, (11, 11))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lowerB , upperB)
        thresh = cv2.erode(thresh, kernel, iterations=3)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        
        # Get contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x1, y1), radius) = cv2.minEnclosingCircle(c)
            # (x,y,w,h) = cv2.boundingRect(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # Draw box on frame
            cv2.rectangle(frame, (int(x1-radius),int(y1-radius)),(int(x1+radius),int(y1+radius)),(0,255,0), 2)
            # cv2.circle(frame, (int(x1), int(y1)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Append position
            position = np.append(position, np.array([[x1, -y1+H]]), axis=0)
        else:
            # Append -1 when no contour
            position = np.append(position, np.array([[-1, -1]]), axis=0)
        
        if show:
            # Show frames
        #    cv2.imshow('thresh', thresh)
            cv2.imshow('frame', frame)
        
        print('Frame [%d/%d]' % (cntFrame, numFrame))
        
        if cntFrame == 1:
            imageList.append(frame)
        elif cntFrame % imageStep == 0:
            image = frame.copy()
            mask = np.zeros(frame.shape,np.uint8)
            mask[int(y1-radius-2):int(y1+radius+2),int(x1-radius-2):int(x1+radius+2),:] = image[int(y1-radius-2):int(y1+radius+2),int(x1-radius-2):int(x1+radius+2),:]
            imageList.append(mask)
        
        # Interupt window by pressing q
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("q")
            break
        
    elapsed = time.time() - start
    print("Time elapsed: %.3f s" % elapsed)
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return position, imageList


if __name__ == '__main__':
    
    position, imageList = BallTracking('[video directory]', show=True)

    plt.figure()
    plt.plot(position[:,0],position[:,1],':k',linewidth=1)
    plt.show()
    
    # Overlay basketball trajectory
    for i,_ in enumerate(imageList):
        if i == 0: 
            outImage = imageList[i]
            continue
        outImage = cv2.addWeighted(outImage,0.9,imageList[i],1,0)

    cv2.imshow('Image', outImage)
    cv2.waitKey(0)

