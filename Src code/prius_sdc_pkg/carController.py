#!/usr/bin/env python3

from .Detection.Lanes.laneDetection import laneDetection
from .Detection.Signs.sign_detection import sign_detect
from .Detection.Lanes.lane_detection import Line, thresh_pipeline, mtx, dist, lane_finding
import cv2
import numpy as np
from numpy import interp
from collections import deque

class Controller():
    def __init__(self):
        self.prev_Mode = "Detection"
        self.prev_Mode_LT = "Detection"
        self.car_speed = 1.0
        # right < 0, left > 0
        self.angle = 0.0
        self.Left_turn_iterations = 0
        self.Frozen_Angle = 0
        self.Detected_LeftTurn = False
        self.Activat_LeftTurn = False        

        self.TrafficLight_iterations = 0
        self.GO_MODE_ACTIVATED = False
        self.STOP_MODE_ACTIVATED = False

        self.angle_queue = deque(maxlen=10)

    def lane_follow(self, Max_Sane_dist, distance, curvature, Mode, Tracked_class):
        IncreaseTireSpeedInTurns = False

        # if((Tracked_class!=0) and (self.prev_Mode == "Tracking") and (Mode == "Detection")):
        #     if(Tracked_class =="speed_sign_60"):
        #         self.car_speed = 1.5
        #     elif(Tracked_class =="stop"):
        #         self.car_speed = 0.0
        if(Tracked_class =="stop"):
            return 0.0, 0.0
            
        self.prev_Mode = Mode # Set prevMode to current Mode

        Max_turn_angle_neg = -1.0
        Max_turn_angle = 1.0
        CarTurn_angle = 0.0
        
        if((distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist) )):
            if(distance > Max_Sane_dist):
                CarTurn_angle = Max_turn_angle + curvature

            else:
                CarTurn_angle = Max_turn_angle_neg + curvature
        else:
            # Turn_angle_interpolated = interp(distance,[-Max_Sane_dist,Max_Sane_dist],[-90,90])
            # CarTurn_angle = (0.65*Turn_angle_interpolated) + (0.35*curvature)
            CarTurn_angle = float(distance / 10)

        if( (CarTurn_angle > Max_turn_angle) or (CarTurn_angle < (-1 *Max_turn_angle) ) ):
            if(CarTurn_angle > Max_turn_angle):
                CarTurn_angle = Max_turn_angle
            else:
                CarTurn_angle = Max_turn_angle_neg

        angle = CarTurn_angle

        # curr_speed = self.car_speed
        
        # if (IncreaseTireSpeedInTurns and (Tracked_class !="left_turn")):
        #     if(angle>30):
        #         car_speed_turn = interp(angle,[30,45],[80,100])
        #         curr_speed = car_speed_turn
        #     elif(angle<-30):
        #         car_speed_turn = interp(angle,[-45,-30],[100,80])
        #         curr_speed = car_speed_turn

        print(self.car_speed)
        return angle , self.car_speed

    def Obey_LeftTurn(self,Angle,Speed,Mode,Tracked_class):
        
        if (Tracked_class == "left_turn"):
            
            Speed = 50

            if ( (self.prev_Mode_LT =="Detection") and (Mode=="Tracking")):
                self.prev_Mode_LT = "Tracking"
                self.Detected_LeftTurn = True

            elif ( (self.prev_Mode_LT =="Tracking") and (Mode=="Detection")):
                self.Detected_LeftTurn = False
                self.Activat_LeftTurn = True

                if ( ((self.Left_turn_iterations % 20 ) ==0) and (self.Left_turn_iterations>100) ):
                    self.Frozen_Angle = self.Frozen_Angle -7 # Move left by 1 degree 
                if(self.Left_turn_iterations==250):
                    self.prev_Mode_LT = "Detection"
                    self.Activat_LeftTurn = False
                    self.Left_turn_iterations = 0
                self.Left_turn_iterations = self.Left_turn_iterations + 1

                if (self.Activat_LeftTurn or self.Detected_LeftTurn):
                    #Follow previously Saved Route
                    Angle = self.Frozen_Angle
        return Angle,Speed,self.Detected_LeftTurn,self.Activat_LeftTurn

    def drive_car(self,Current_State, Inc_LT):
        [Distance, Curvature, img, Mode, Tracked_classes] = Current_State

        current_speed = 0

        if((Distance != -1000) and (Curvature != -1000)):
            self.angle, current_speed = self.lane_follow(img.shape[1]/2, Distance, Curvature, Mode, Tracked_classes)

        self.angle_queue.append(self.angle)
        self.angle = (sum(self.angle_queue)/len(self.angle_queue))

        if(Inc_LT):
            self.angle,current_speed, Detected_LeftTurn, Activat_LeftTurn = self.Obey_LeftTurn(self.angle, current_speed, Mode, Tracked_classes) 
        else:
            Detected_LeftTurn = False
            Activat_LeftTurn = False

        return self.angle,current_speed, Detected_LeftTurn, Activat_LeftTurn 

class Car():
    def __init__( self, Inc_LT = True ):
        
        self.Control = Controller()
        self.Inc_LT = Inc_LT
        self.Tracked_class = "Unknown"

    def display_state(self,frame_disp,angle_of_car,current_speed):    
        ###################################################  Displaying CONTROL STATE ####################################

        if (angle_of_car <-10):
            direction_string="[ Left ]"
            color_direction=(120,0,255)
        elif (angle_of_car >10):
            direction_string="[ Right ]"
            color_direction=(120,0,255)
        else:
            direction_string="[ Straight ]"
            color_direction=(0,255,0)

        if(current_speed>0):
            direction_string = "Moving --> "+ direction_string
        else:
            color_direction=(0,0,255)


        cv2.putText(frame_disp,str(direction_string),(20,40),cv2.FONT_HERSHEY_DUPLEX,0.4,color_direction,1)

        angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + "deg ," + str(int(current_speed*40)) + "mph ]"
        cv2.putText(frame_disp,str(angle_speed_str),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)

    def car_driving(self, frame):
        # image = frame[330:,200:950]
        # image = frame.copy()

        # cv2.imshow("image", frame)

        img_undist = cv2.undistort(frame, mtx, dist, None, mtx)
        img_thresh = thresh_pipeline(img_undist, gradx_thresh=(25,255), grady_thresh=(10,255), s_thresh=(100, 255), v_thresh=(0, 255))
        
        lane_l = Line()
        lane_r = Line()
        lane_l.detected = False
        lane_r.detected = False
        img_out, distance, curvature = lane_finding(frame)
        Mode, Tracked_classes = sign_detect(frame)

        self.Tracked_class = Tracked_classes

        Current_state = [distance, curvature, img_out, Mode, Tracked_classes]

        Angle,Speed, Detected_LeftTurn, Activat_LeftTurn  = self.Control.drive_car(Current_state, self.Inc_LT)

        # self.display_state(img_out, self.Control.angle, self.Control.car_speed)
        
        # print(Speed, Angle)
        # print(distance)


        return Angle, Speed, img_out
        # return 0.0, 0.0, img_out

    

        # image = cv2.resize(image, (320, 240))
        # cv2.imshow("raw_raw", image)
"""
********************************************************************************************
********************************************************************************************
********************** LANE DETECTION ******************************************************
""" 
        

"""
********************************************************************************************
********************************************************************************************
"""

"""
********************************************************************************************
********************************************************************************************
********************** SIGN DETECTION ******************************************************
"""

"""
********************************************************************************************
********************************************************************************************
"""
