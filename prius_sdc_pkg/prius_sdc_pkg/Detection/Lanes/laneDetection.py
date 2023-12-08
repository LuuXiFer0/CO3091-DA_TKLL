#!/usr/bin/env python3

from .Stage1_Segmentation.colorSegmentation import lane_segmentation
from .Stage2_Estimation.midlane_estimation import midlane_estimation
from .Stage3_Cleaning.outerlane_extraction import extract_leftside_outerlane, lane_extension
from .Stage4_DataExtraction.data_info import FetchInfoAndDisplay

from ...config import config
import cv2

def laneDetection(image):
    cropped_image = image[160 : , :]
    cv2.imshow("cropped", cropped_image)
    # cropped_image = cv2.resize(cropped_image, (320, 240))

    # cropped_image = cv2.resize(cropped_image, (320, 240))
    mid_lane_mask, mid_lane_edge, outer_lane_edge, outer_lane_sidesep, outer_lane_point = lane_segmentation(cropped_image, config.minArea_resized)
    estimated_midlane = midlane_estimation(mid_lane_mask, config.MaxDist_resized)
    outer_lane, outer_cnt, mid_cnt , offset = extract_leftside_outerlane(estimated_midlane, outer_lane_sidesep, outer_lane_point)
    extended_midlane, extended_outerlane = lane_extension(estimated_midlane, outer_lane, mid_cnt, outer_cnt)
    Distance , Curvature = FetchInfoAndDisplay(mid_lane_mask,extended_midlane,extended_outerlane,cropped_image,offset)


    # cv2.imshow("mid_lane_mask", mid_lane_mask)
    # cv2.imshow("mid_lane_edge", mid_lane_edge)
    # cv2.imshow("lane1", lane1)
    # cv2.imshow("outer_lane_edge", outer_lane_edge)
    cv2.imshow("outer_lane_sidesep", outer_lane_sidesep)

    #print(len(outer_lane_point))
    # cv2.imshow("Estimation", estimated_midlane)
    # cv2.imshow("outer lane", outer_lane)
    # cv2.imshow("extended midlane", extended_midlane)
    # cv2.imshow("extended_outerlane", extended_outerlane)
    # print(image.shape)
    # cv2.imshow("ra", cropped_image)
    # print(extended_midlane.shape, extended_outerlane.shape, cropped_image.shape)
    #cv2.imshow("outer cnt", outer_cnt)
    # cv2.waitKey(0)
    return Distance, Curvature
