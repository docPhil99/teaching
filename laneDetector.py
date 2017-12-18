#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:42:02 2017

@author: phil
"""

import PMB.common.FastVideo as fv
import cv2
import numpy as np
fvs = fv.FastVideoReader("/media/phil/Seagate Expansion Drive/oldCdrive/DashCamvideo/FirstTest/RE2_0007.MOV").start()

res = 0
while fvs.more() and not res==27:
    frame = fvs.read()
    frame=cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA)
    cv2.imshow("Frame", frame)
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, length, _ = frame.shape
    bottom_left_corner = (0, height)                                      # (1)
    bottom_right_corner = (length, height)
    center = (int(length / 2), int(height/2))
    region = [np.array([bottom_left_corner,center,bottom_right_corner])]  # (2)
    mask = np.zeros_like(grayscale_img)                            # (1)
    keep_region_color = 255
    ignore_mask_color=0
    cv2.fillPoly(mask, region, keep_region_color)                  # (2)
    region_selected_image = cv2.bitwise_and(grayscale_img, mask)   # (3)
    cv2.imshow("region_selected_image",  mask)
    kernel_size = 3
    blurred =  cv2.GaussianBlur(region_selected_image, (kernel_size, kernel_size), 0)


    low_threshold = 150
    high_threshold = 300    
    canny_transformed = cv2.Canny(blurred, low_threshold, high_threshold)
    cv2.imshow(" canny_transformed",   canny_transformed)
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    Line = namedtuple("Line", "x1 y1 x2 y2")
    lines = [Line(*line[0]) for line in lines]   
    
    res=cv2.waitKey(1) 

 
    