#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:19:18 2020

@author: Jiale Hu
"""

import cv2
import numpy as np

img = cv2.imread("[image directory]")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

B = [np.mean(img[:,:,0]), np.min(img[:,:,0]), np.max(img[:,:,0])]
G = [np.mean(img[:,:,1]), np.min(img[:,:,1]), np.max(img[:,:,1])]
R = [np.mean(img[:,:,2]), np.min(img[:,:,2]), np.max(img[:,:,2])]

H = [np.mean(hsv[:,:,0]), np.min(hsv[:,:,0]), np.max(hsv[:,:,0])]
S = [np.mean(hsv[:,:,1]), np.min(hsv[:,:,1]), np.max(hsv[:,:,1])]
V = [np.mean(hsv[:,:,2]), np.min(hsv[:,:,2]), np.max(hsv[:,:,2])]
