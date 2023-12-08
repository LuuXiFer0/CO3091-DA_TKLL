import numpy as np
import math
import cv2
from ..helpful_function import rearrange_contour

def findlaneCurvature(x1,y1,x2,y2):
    offset_Vert=90# angle found by tan-1 (slop) is wrt horizontal --> This will shift to wrt Vetical

    if((x2-x1)!=0):
        slope = (y2-y1)/(x2-x1)
        y_intercept = y2 - (slope*x2) #y= mx+c
        anlgeOfinclination = math.atan(slope) * (180 / np.pi)#Conversion to degrees
    else:
        slope=1000#infinity
        y_intercept=0#None [Line never crosses the y axis]

        anlgeOfinclination = 90#vertical line

        #print("Vertical Line [Undefined slope]")
    if(anlgeOfinclination!=90):
        if(anlgeOfinclination<0):#right side
            angle_wrt_vertical = offset_Vert + anlgeOfinclination
        else:#left side
            angle_wrt_vertical = anlgeOfinclination - offset_Vert
    else:
        angle_wrt_vertical= 0#aligned
    return angle_wrt_vertical

def EstimateNonMidMask(MidEdgeROi):
    Mid_Hull_Mask = np.zeros((MidEdgeROi.shape[0], MidEdgeROi.shape[1], 1), dtype=np.uint8)
    contours = cv2.findContours(MidEdgeROi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    if (len(contours) > 0):
        hull_list = []
        contours = np.concatenate(contours)
        hull = cv2.convexHull(contours)
        hull_list.append(hull)
        # Draw contours + hull results
        Mid_Hull_Mask = cv2.drawContours(Mid_Hull_Mask, hull_list, 0, 255, -1)
    Non_Mid_Mask = cv2.bitwise_not(Mid_Hull_Mask)
    return Non_Mid_Mask



def LanePoints(MidLane,OuterLane,Offset_correction):
	Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	Outer_cnts = cv2.findContours(OuterLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	if(Mid_cnts and Outer_cnts):
		Mid_cnts_Rowsorted = rearrange_contour(Mid_cnts,"rows")
		Outer_cnts_Rowsorted = rearrange_contour(Outer_cnts,"rows")
		#print(Mid_cnts_Rowsorted)
		Mid_Rows = Mid_cnts_Rowsorted.shape[0]
		Outer_Rows = Outer_cnts_Rowsorted.shape[0]

		Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:]
		Mid_highP = Mid_cnts_Rowsorted[0,:]
		Outer_lowP = Outer_cnts_Rowsorted[Outer_Rows-1,:]
		Outer_highP = Outer_cnts_Rowsorted[0,:]

		LanePoint_lower = ( int( (Mid_lowP[0] + Outer_highP[0]  ) / 2 ) + Offset_correction, int( (Mid_lowP[1]  + Outer_highP[1] ) / 2 ) )
		LanePoint_top   = ( int( (Mid_highP[0] + Outer_lowP[0]) / 2 ) + Offset_correction, int( (Mid_highP[1] + Outer_lowP[1]) / 2 ) )
		# print(Mid_lowP, Outer_lowP, Mid_highP, Outer_highP)
		return LanePoint_lower,LanePoint_top
	else:
		return (0,0),(0,0)
	
def FetchInfoAndDisplay(Mid_lane_edge,Mid_lane,Outer_Lane,frame,Offset_correction):
	Traj_lowP,Traj_upP = LanePoints(Mid_lane,Outer_Lane,Offset_correction)
    
	PerpDist_LaneCentralStart_CarNose= -1000
	if(Traj_lowP!=(0,0)):
		PerpDist_LaneCentralStart_CarNose = Traj_lowP[0] - int(Mid_lane.shape[1]/2)
	curvature = findlaneCurvature(Traj_lowP[0],Traj_lowP[1],Traj_upP[0],Traj_upP[1])
	Mid_lane_edge = cv2.bitwise_and(Mid_lane_edge,Mid_lane)
	Lanes_combined = cv2.bitwise_or(Outer_Lane,Mid_lane)

	ProjectedLane = np.zeros(Lanes_combined.shape,Lanes_combined.dtype)
	cnts = cv2.findContours(Lanes_combined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

	if (cnts):
		cnts = np.concatenate(cnts)
		cnts = np.array(cnts)
		cv2.fillConvexPoly(ProjectedLane, cnts, 255)

	Mid_less_Mask = EstimateNonMidMask(Mid_lane_edge)
	ProjectedLane = cv2.bitwise_and(Mid_less_Mask,ProjectedLane)

		# 8. Draw projected lane
	Lane_drawn_frame = frame

	Lane_drawn_frame[ProjectedLane==255] = Lane_drawn_frame[ProjectedLane==255] + (0,100,0)
	Lane_drawn_frame[Outer_Lane==255] = Lane_drawn_frame[Outer_Lane==255] + (0,0,100)# Outer Lane Coloured Red
	Lane_drawn_frame[Mid_lane==255] = Lane_drawn_frame[Mid_lane==255] + (100,0,0)# Mid Lane Coloured Blue
	Out_image = Lane_drawn_frame

	cv2.line(Out_image,(int(Out_image.shape[1]/2),Out_image.shape[0]),(int(Out_image.shape[1]/2),Out_image.shape[0]-int (Out_image.shape[0]/5)),(0,0,255),2)
	cv2.line(Out_image,Traj_lowP,Traj_upP,(255,0,0),2)

	if(Traj_lowP!=(0,0)):
		cv2.line(Out_image,Traj_lowP,(int(Out_image.shape[1]/2),Traj_lowP[1]),(255,255,0),2)# distance of car center with lane path

	curvature_str="Curvature = " + f"{curvature:.2f}"
	PerpDist_ImgCen_CarNose_str="Distance = " + str(PerpDist_LaneCentralStart_CarNose)
	textSize_ratio = 0.5
	cv2.putText(Out_image,curvature_str,(10,30),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
	cv2.putText(Out_image,PerpDist_ImgCen_CarNose_str,(10,50),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(255,0,255),1)
	# print(Traj_lowP, Traj_upP)
	cv2.imshow("combi",Lanes_combined)
      
	print(PerpDist_LaneCentralStart_CarNose,curvature)
	return PerpDist_LaneCentralStart_CarNose,curvature
      

