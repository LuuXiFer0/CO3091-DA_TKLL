import cv2
import numpy as np

def detect_large_objects(grayscale_img, minArea):
    """
    Detect and filter large objects in a grayscale image.

    Args:
        grayscale_img (numpy.ndarray): Input grayscale image.
        minArea (int): Minimum area for retaining detected objects.

    Returns:
        numpy.ndarray: Thresholded image with small objects removed.
    """
    # Threshold the input grayscale image
    thresh = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY)[1]

    # Detect contours and hierarchy
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    cnt_small_obj = []
    for idx, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if (area < minArea):
            cnt_small_obj.append(cnt)

    # Draw contours to remove small objects
    thresh = cv2.drawContours(thresh, cnt_small_obj, -1, 0, -1)
    return thresh


def FindExtremas(img):
    positions = np.nonzero(img) # position[0] 0 = rows 1 = cols
    if (len(positions)!=0):
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        return top,bottom
    else:
        return 0,0

def FindLowestRow(img):
    positions = np.nonzero(img) # position[0] 0 = rows 1 = cols
    
    if (len(positions)!=0):
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        return bottom
    else:
        return img.shape[0]
    
def ret_largest_contour_outer_lane(gray,minArea):
    LargestContour_Found = False
    thresh=np.zeros(gray.shape,dtype=gray.dtype)
    _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    
    #################################### TESTING SHADOW BREAKER CODE BY DILATING####################
    # 3. Dilating Segmented ROI's
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    bin_img_dilated = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)    #Find the two Contours for which you want to find the min distance between them.
    bin_img_ret = cv2.morphologyEx(bin_img_dilated, cv2.MORPH_ERODE, kernel)    #Find the two Contours for which you want to find the min distance between them.
    bin_img = bin_img_ret
    #################################### TESTING SHADOW BREAKER CODE BY DILATING####################

    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    Max_Cntr_area = 0
    Max_Cntr_idx= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > Max_Cntr_area:
            Max_Cntr_area = area
            Max_Cntr_idx = index
            LargestContour_Found = True
    
    if Max_Cntr_area < minArea:
        LargestContour_Found = False
    if ((Max_Cntr_idx!=-1) and (LargestContour_Found)):
        thresh = cv2.drawContours(thresh, cnts, Max_Cntr_idx, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
    return thresh, LargestContour_Found

def ROI_extracter(image,strtPnt,endPnt):
    #  Selecting Only ROI from Image
    ROI_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(ROI_mask,strtPnt,endPnt,255,thickness=-1)
    #image_ROI = cv2.bitwise_and(image,image,mask=ROI_mask)
    image_ROI = cv2.bitwise_and(image,ROI_mask)
    return image_ROI

def ExtractPoint(img,specified_row):
    Point= (0,specified_row)
    specified_row_data = img[ specified_row-1,:]
    #print("specified_row_data",specified_row_data)
    positions = np.nonzero(specified_row_data) # position[0] 0 = rows 1 = cols
    #print("positions",positions)    
    #print("len(positions[0])",len(positions[0]))    
    if (len(positions[0])!=0):
        #print(positions[0])
        min_col = positions[0].min()
        Point=(min_col,specified_row)
    return Point


def ret_lowest_edge_points(gray):
    
    Outer_Points_list=[]
    thresh = np.zeros(gray.shape,dtype=gray.dtype)
    Lane_OneSide=np.zeros(gray.shape,dtype=gray.dtype)
    Lane_TwoSide=np.zeros(gray.shape,dtype=gray.dtype)

    _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    thresh = cv2.drawContours(thresh, cnts, 0, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
    # Boundary of the Contour is extracted and Saved in Thresh

    Top_Row,Bot_Row = FindExtremas(thresh)

    Contour_TopBot_PortionCut = ROI_extracter(thresh,(0, Top_Row + 5),(thresh.shape[1],Bot_Row-5))

    cnts2 = cv2.findContours(Contour_TopBot_PortionCut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    LowRow_a=-1
    LowRow_b=-1
    
    Euc_row=0# Row for the points to be compared

    First_line = np.copy(Lane_OneSide)
    cnts_tmp = []
    

    if(len(cnts2)>1):
        for index_tmp, cnt_tmp in enumerate(cnts2):
            if((cnt_tmp.shape[0])>50):
                cnts_tmp.append(cnt_tmp)
        cnts2 = cnts_tmp

    for index, cnt in enumerate(cnts2):
        Lane_OneSide = np.zeros(gray.shape,dtype=gray.dtype)
        Lane_OneSide = cv2.drawContours(Lane_OneSide, cnts2, index, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
        Lane_TwoSide = cv2.drawContours(Lane_TwoSide, cnts2, index, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]

        if(len(cnts2)==2):
            if (index==0):
                First_line = np.copy(Lane_OneSide)
                LowRow_a = FindLowestRow(Lane_OneSide)
            elif(index==1):
                LowRow_b = FindLowestRow(Lane_OneSide)
                if(LowRow_a<LowRow_b):# First index is shorter 
                    Euc_row=LowRow_a
                else:
                    Euc_row=LowRow_b
                #print("Euc_row",Euc_row)
                #cv2.namedWindow("First_line",cv2.WINDOW_NORMAL)
                #cv2.imshow("First_line",First_line)
                #cv2.waitKey(0)
                Point_a = ExtractPoint(First_line,Euc_row)
                Point_b = ExtractPoint(Lane_OneSide,Euc_row)
                Outer_Points_list.append(Point_a)
                Outer_Points_list.append(Point_b)
    
    return Lane_OneSide, Lane_TwoSide, Outer_Points_list