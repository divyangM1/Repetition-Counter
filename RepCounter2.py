import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    if angle > 180.0:
        angle = 360-angle

    return angle
    
stage = False
rep = 0
p = 0
def ans():
    global rep
    global p
    global angled
    i = 0
    val = 0
    while i<8:
        if angled > 140.0 or i > 7:
            p = 0
            break
        time.sleep(1)
        i+=1
        p+=1
        
    if i == 8:
        rep+=1
        p = 0




# Setting up video instance for mediapipe
cam = cv2.VideoCapture('KneeBendVideo2.mp4')
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cam.isOpened():
        ret, frame = cam.read()
        
        # BGR to RGB so that mediapipe can detect
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Detection
        result = pose.process(image)
        
        # RGB to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Position Pointers
        try:
            position =  result.pose_landmarks.landmark
            
            
            # 3 point coordinates to calculate angle
            
            hip = [position[mp_pose.PoseLandmark.LEFT_HIP.value].x,position[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [position[mp_pose.PoseLandmark.LEFT_KNEE.value].x,position[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [position[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,position[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # get angle
            angled = angle(hip,knee,ankle)
            
            
            #present
            cv2.putText(image, str(angled), tuple(np.multiply(knee, [640, 480]).astype(int)),
                                                  cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
            
#             a,b = calc_time(angled)
#             x = a
#             y = b
            if angled > 140.0:
                stage = True
            if angled < 140.0 and stage == True:
                Thread(target = ans).start()
                stage = False
                
            print(rep, p,angled)
            cv2.putText(image, "Rep: "+str(rep), (49,93),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(image, "Seconds: "+str(p), (49,123),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            
                

            
            
        except:
            pass
        
        #SHow detections
        
        mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow("mediapipe view",image)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()