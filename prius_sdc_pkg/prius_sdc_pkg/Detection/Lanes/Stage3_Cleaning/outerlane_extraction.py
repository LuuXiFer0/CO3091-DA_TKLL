import cv2
import numpy as np
from ..helpful_function import distance, rearrange_contour

def detect_crossing_midlane(midlane, midlane_contour, outerlane_contour):
    midlane_copy = midlane.copy()
    Ref_To_Path_Image = np.zeros_like(midlane)

    sorted_mid_cnt = rearrange_contour(midlane_contour, "row")
    sorted_outer_cnt = rearrange_contour(outerlane_contour, "row")

    mid_lowest_point = sorted_mid_cnt[sorted_mid_cnt.shape[0] - 1, :]
    outer_lowest_point = sorted_outer_cnt[sorted_outer_cnt.shape[0] - 1, :]

    center_traj = (int((mid_lowest_point[0] + outer_lowest_point[0]) / 2), int((mid_lowest_point[1] + outer_lowest_point[1]) / 2))

    cv2.line(midlane_copy, tuple(mid_lowest_point), (mid_lowest_point[0], midlane_copy.shape[0]-1 ), (255, 255, 0), 2)# distance of car center with lane path
    cv2.line(Ref_To_Path_Image, center_traj, (int(Ref_To_Path_Image.shape[1]/2), Ref_To_Path_Image.shape[0]), (255, 255, 0), 2)# distance of car center with lane path

    left_handside = ( int(Ref_To_Path_Image.shape[1]/2) - center_traj[1] ) > 0

    if(np.any(cv2.bitwise_and(Ref_To_Path_Image, midlane_copy) > 0)):
        return True, left_handside
    else: 
        return False, left_handside


def extract_leftside_outerlane(midlane, outerlane, outerlane_point):
    return_outer_lane = np.zeros(outerlane.shape, outerlane.dtype)

    midlane_contour, _ = cv2.findContours(midlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outerlane_contour, _ = cv2.findContours(outerlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not outerlane_contour:
        isnot_found = True
    else:
        isnot_found = False

    ref_point = (0,0)
    if(midlane_contour):
        ref_point = midlane_contour[0][0][0]


    # print(len(outerlane_contour))
    if(midlane_contour):
        if(len(outerlane_point) == 2):
            point_a = outerlane_point[0]
            point_b = outerlane_point[1]

            closer_side_idx = 0
            if(distance(ref_point, point_a) <= distance(ref_point, point_b)):
                closer_side_idx = 0
            elif(len(outerlane_contour) > 1):
                closer_side_idx = 1
            
            crossing_mid, _ = detect_crossing_midlane(midlane, midlane_contour, outerlane_contour)

            return_outer_lane = cv2.drawContours(return_outer_lane, outerlane_contour, closer_side_idx, (255, 255, 255), 1)
            return_outer_contour = [outerlane_contour[closer_side_idx]]

            if(not crossing_mid):
                return return_outer_lane, return_outer_contour, midlane_contour, 0
            else:
                outerlane = np.zeros_like(outerlane)

        if(not np.any(outerlane > 0)):
            sorted_mid_cnt = rearrange_contour(midlane_contour, "row")
            mid_lowest_point = sorted_mid_cnt[sorted_mid_cnt.shape[0] - 1, :]
            mid_highest_point = sorted_mid_cnt[0, :]

            right_line = False
            if(isnot_found):
                if(mid_lowest_point[0] < int(midlane.shape[1]/2)):
                    right_line = True
            else:
                if(crossing_mid):
                    right_line = True

            if(right_line):
                offset = 20
                low_pt_col = high_pt_col = midlane.shape[1] - 1
            else:
                offset = -20
                low_pt_col = high_pt_col = 0
            
            outerlane = cv2.line(outerlane, (low_pt_col, mid_lowest_point), (high_pt_col, mid_highest_point), (255, 255, 255), 1)
            outerlane_contour, _ = cv2.findContours(outerlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            return outerlane, outerlane_contour, midlane_contour, offset
        
    return outerlane, outerlane_contour, midlane_contour, 0

def lane_extension(midlane, outerlane, midlane_contour, outerlane_contour):
    if(midlane.any() and outerlane.any()):
        sorted_mid_cnt = rearrange_contour(midlane_contour, "row")
        sorted_outer_cnt = rearrange_contour(outerlane_contour, "row")
        mid_lowest_point = sorted_mid_cnt[sorted_mid_cnt.shape[0] - 1, :]
        outer_lowest_point = sorted_outer_cnt[sorted_outer_cnt.shape[0] - 1, :]
        img_bottom = midlane.shape[0]

        if(mid_lowest_point[1] < img_bottom):
            cv2.line(midlane, tuple(mid_lowest_point), (mid_lowest_point[0], img_bottom), (255, 255, 255), 2)

        if(outer_lowest_point[1] < img_bottom):
            if(sorted_outer_cnt.shape[0] > 20):
                shift = 20
            else:
                shift = 2
            outer_last_10pt = sorted_outer_cnt[sorted_outer_cnt.shape[0] - shift : sorted_outer_cnt.shape[0] - 1 : 2, :]

            if(len(outer_last_10pt) > 1):
                x_line = outer_last_10pt[:, 0]
                y_line = outer_last_10pt[:, 1]
                fit_line = np.polyfit(x_line, y_line, 1)
                slope = fit_line[0]
                intercept = fit_line[1]

                if(slope < 0):
                    cross_point_col = 0
                    cross_point_row = intercept
                else:
                    cross_point_col = outerlane.shape[1] - 1
                    cross_point_row = cross_point_col * slope + intercept
                outerlane = cv2.line(outerlane, (cross_point_col, int(cross_point_row)), tuple(outer_lowest_point), (255, 255, 255), 1)

                if(cross_point_row < img_bottom):
                    outerlane = cv2.line(outerlane, (cross_point_col, int(cross_point_row)), (cross_point_col, img_bottom), (255, 255, 255), 1)
    return midlane, outerlane 



        
