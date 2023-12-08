#!/usr/bin/env python3
import cv2
import numpy as np
import math

def distance(a, b):
    return math.sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)

def approx_distance_centr(cnt, cnt_flag):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid_a = (cX, cY)

    M_flag = cv2.moments(cnt_flag)
    cX_flag = int(M_flag["m10"] / M_flag["m00"])
    cY_flag = int(M_flag["m01"] / M_flag["m00"])
    centroid_b = (cX_flag, cY_flag)

    centroid_distance = distance(centroid_a, centroid_b)
    return centroid_distance, centroid_a, centroid_b

def cover_largest_contour(img_gray):
    is_found = False
    thresh = np.zeros(img_gray.shape, dtype = img_gray.dtype)
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    max_contour_area = 0
    max_contour_idx = -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > max_contour_area:
            max_contour_area = area
            max_contour_idx = index
            is_found = True
    if (max_contour_idx!=-1):
        thresh = cv2.drawContours(thresh, cnts, max_contour_idx, (255, 255, 255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
        
    return thresh, is_found

def midlane_estimation(midlane_patches, max_distance):
    image_draw = cv2.cvtColor(midlane_patches, cv2.COLOR_GRAY2BGR)

    cnts = cv2.findContours(midlane_patches, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    min_area = 1
    cnts_remain = []
    for _, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if(area > min_area):
            cnts_remain.append(cnt)
    cnts = cnts_remain

    trajectory_idx = []
    for idx, cnt in enumerate(cnts):
        min_centroid_distance = 100000
        min_centr_a = 0
        min_centr_b = 0
        min_centr_idx = 0
        for idx_flag in range(idx, len(cnts)):
            cnt_flag = cnts[idx_flag]
            if(idx != idx_flag):
                centroid_distance , centroid_a, centroid_b = approx_distance_centr(cnt, cnt_flag)

                if(centroid_distance < min_centroid_distance):
                    if(len(trajectory_idx) == 0):
                        min_centroid_distance = centroid_distance
                        min_centr_a = centroid_a
                        min_centr_b = centroid_b
                        min_centr_idx = idx_flag
                    else:
                        recent = False
                        for i in range(len(trajectory_idx)):
                            if ( (idx_flag == i) and (idx == trajectory_idx[i]) ):
                                recent= True
                        if not recent:
                            min_centroid_distance = centroid_distance
                            min_centr_a = centroid_a
                            min_centr_b = centroid_b
                            min_centr_idx = idx_flag
        if(min_centroid_distance > max_distance): 
            break
        if(min_centr_a != 0):
            trajectory_idx.append(min_centr_idx)
            cv2.line(image_draw, min_centr_a, min_centr_b, (0, 255, 0), 2)

    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2GRAY)
    estimated_midlane, largest_found = cover_largest_contour(image_draw)

    if largest_found:
        return estimated_midlane
    else:
        return midlane_patches

