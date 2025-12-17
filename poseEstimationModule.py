import mediapipe as mp
import cv2
import math 

class PoseEstimation():
    def __init__(self, mode = False, upper_body = False, smooth_landmark = True, min_det_con = 0.5, min_track_con = 0.5):
        self.mode = mode
        self.smooth_landmark = smooth_landmark
        self.upper_body = upper_body
        self.min_det_con = min_det_con
        self.min_track_con = min_track_con
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpdraw = mp.solutions.drawing_utils
        
    def findPose(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw :
                self.mpdraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                
        return frame
                
    def positions(self, frame, draw = True):
        
        self.lmlist = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    self.lmlist.append([id, cx, cy])
                    if draw :
                        cv2.circle(frame, (cx, cy), 8, (255, 255, 0), cv2.FILLED)
                    
        return self.lmlist
    
    def findAngle(self, img, i1, i2, i3, draw = True):
        x1, y1 = self.lmlist[i1][1:]
        x2, y2 = self.lmlist[i2][1:]
        x3, y3 = self.lmlist[i3][1:]
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle <=0:
            angle += 360
        #print(angle)
        
        if draw :
            cv2.line(img, (x1, y1),(x2, y2), (0, 255, 255), 3)
            cv2.line(img, (x2, y2),(x3, y3), (0, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 5)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 5)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 5)
        return angle 
        