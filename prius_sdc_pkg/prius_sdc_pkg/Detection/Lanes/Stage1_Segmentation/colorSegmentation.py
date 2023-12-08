#!/usr/bin/env python3

import cv2 
import numpy as np
from ....config import config
from ..morphOp import detect_large_objects, ret_largest_contour_outer_lane, ret_lowest_edge_points

# Define Constants
HLS = 0

WHITE_LOWER_HUE = 0
WHITE_LOWER_LIGHT = 225
WHITE_LOWER_SAT = 0

YELLOW_LOWER_HUE = 20
YELLOW_UPPER_HUE = 33
YELLOW_LOWER_LIGHT = 120
YELLOW_LOWER_SAT = 0

def color_segmentation(hls_image, lower_range, upper_range):
    """Apply color segmentation to an HLS image.

    Args:
        hls_image (numpy.ndarray): The input HLS image.
        lower_range (tuple): Lower range values for color segmentation.
        upper_range (tuple): Upper range values for color segmentation.

    Returns:
        numpy.ndarray: The segmented mask.
    """
    lower = np.array(lower_range, dtype=np.uint8)
    upper = np.array(upper_range, dtype=np.uint8)

    mask_in_range = cv2.inRange(hls_image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.morphologyEx(mask_in_range, cv2.MORPH_DILATE, kernel)
    return mask_dilated

def mask_extract():
    global HLS, src
    white_mask = color_segmentation(HLS, (WHITE_LOWER_HUE, WHITE_LOWER_LIGHT, WHITE_LOWER_SAT), (255, 255, 255))
    yellow_mask = color_segmentation(HLS, (YELLOW_LOWER_HUE, YELLOW_LOWER_LIGHT, YELLOW_LOWER_SAT), (YELLOW_UPPER_HUE, 255, 255))

    yellow_mask_ = yellow_mask != 0
    yellow_dest = src * (yellow_mask_[:, :, None].astype(src.dtype))

    white_mask_ = white_mask != 0
    white_dest = src * (white_mask_[:, :, None].astype(src.dtype))

    if config.debugging_Lane and config.debugging and config.debugging_L_ColorSeg:
        cv2.imshow('white_mask', white_dest)
        cv2.imshow('yellow_mask', yellow_dest)

# Apply mask on created window to defined best range for each color
#WHITE MASK
def change_white_lower_hue(value):
    global white_lower_hue
    white_lower_hue = value
    mask_extract()

def change_white_lower_light(value):
    global white_lower_light
    white_lower_light = value
    mask_extract()

def change_white_lower_sat(value):
    global white_lower_sat
    white_lower_sat = value
    mask_extract()

#YELLOW MASK
def change_yellow_lower_hue(value):
    global YELLOW_LOWER_HUE
    YELLOW_LOWER_HUE = value
    mask_extract()

def change_yellow_upper_hue(value):
    global YELLOW_UPPER_HUE
    YELLOW_UPPER_HUE = value
    mask_extract()

def change_yellow_lower_light(value):
    global YELLOW_LOWER_LIGHT
    YELLOW_LOWER_LIGHT = value
    mask_extract()

def change_yellow_lower_sat(value):
    global YELLOW_LOWER_SAT
    YELLOW_LOWER_SAT = value
    mask_extract()

# cv2.namedWindow("white_range")
# cv2.namedWindow("yellow_range")

# cv2.createTrackbar("white_lower_hue","white_range",WHITE_LOWER_HUE,255,change_white_lower_hue)
# cv2.createTrackbar("white_lower_light","white_range",WHITE_LOWER_LIGHT,255,change_white_lower_light)
# cv2.createTrackbar("white_lower_sat","white_range",WHITE_LOWER_SAT,255,change_white_lower_sat)
# cv2.createTrackbar("yellow_lower_hue","yellow_range",YELLOW_LOWER_HUE,255,change_yellow_lower_hue)
# cv2.createTrackbar("yellow_upper_hue","yellow_range",YELLOW_UPPER_HUE,255,change_yellow_upper_hue)
# cv2.createTrackbar("yellow_lower_light","yellow_range",YELLOW_LOWER_LIGHT,255,change_yellow_lower_light)
# cv2.createTrackbar("yellow_lower_sat","yellow_range",YELLOW_LOWER_SAT,255,change_yellow_lower_sat)

def get_mask_and_edge_of_large_objects(frame, mask, min_area):
    """Get the mask and edge of large objects in the frame.

    Args:
        frame (numpy.ndarray): The input frame.
        mask (numpy.ndarray): The mask to apply to the frame.
        min_area (int): Minimum area threshold for detecting large objects.

    Returns:
        tuple: A tuple containing mask and edge images.
    """
    lane_frame = cv2.bitwise_and(frame, frame, mask=mask)
    lane_frame_gray = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2GRAY)
    mask_of_large_object = detect_large_objects(lane_frame_gray, min_area)
    lane_frame_gray_reduce_noise = cv2.bitwise_and(lane_frame_gray, mask_of_large_object)
    lane_frame_blur = cv2.GaussianBlur(lane_frame_gray_reduce_noise, (5, 5), 0)
    lane_edge = cv2.Canny(lane_frame_blur, 50, 150, None, 3)

    return mask_of_large_object, lane_edge

def white_region_edge_detect(frame, white_range, min_area):
    mask, edge = get_mask_and_edge_of_large_objects(frame, white_range, min_area)
    return mask, edge

def yellow_region_edge_detect(frame, yellow_range, min_area):
    outer_points = []
    mask, edge = get_mask_and_edge_of_large_objects(frame, yellow_range, min_area)
    edges = None
    largest_mask, largest_found = ret_largest_contour_outer_lane(mask, min_area)

    if largest_found:
        largest_edge = cv2.bitwise_and(edge, largest_mask)
        _, lane_side_separate, outer_points = ret_lowest_edge_points(largest_edge)
        edges = largest_edge
    else:
        lane_side_separate = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    return edges, lane_side_separate, outer_points

def lane_segmentation(frame, min_area):
    global HLS, src
    src = frame.copy()
    hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    white_range = color_segmentation(hls_image, (WHITE_LOWER_HUE, WHITE_LOWER_LIGHT, WHITE_LOWER_SAT), (255, 255, 255))
    yellow_range = color_segmentation(hls_image, (YELLOW_LOWER_HUE, YELLOW_LOWER_LIGHT, YELLOW_LOWER_SAT), (YELLOW_UPPER_HUE, 255, 255))

    # cv2.imshow("white_range", white_range)
    # cv2.imshow("yellow_range", yellow_range)

    mid_lane_mask, mid_lane_edge = white_region_edge_detect(frame, white_range, min_area)
    outer_lane_edge, outer_lane_sidesep, outer_lane_point = yellow_region_edge_detect(frame, yellow_range, min_area)

    return mid_lane_mask, mid_lane_edge, outer_lane_edge, outer_lane_sidesep, outer_lane_point