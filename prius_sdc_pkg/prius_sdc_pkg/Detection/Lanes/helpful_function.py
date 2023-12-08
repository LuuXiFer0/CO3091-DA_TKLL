#!/usr/bin/env python3

import cv2 
import numpy as np
import math

def distance(a, b):
    return math.sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)

def rearrange_contour(contour, axis):
    if(contour):
        cnt = contour[0]
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))

        if(axis == "row"):
            sorted_cnt = cnt[np.lexsort((cnt[:, 0], cnt[:, 1]))]
        else:
            sorted_cnt = cnt[np.lexsort((cnt[:, 1], cnt[:, 0]))]
        return sorted_cnt
    else:
        return contour