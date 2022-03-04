import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from math import hypot
import threading

mp_pose = mp.solutions.mediapipe.python.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils 
results = None
def detectPose(image):

    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    global results
    results = pose.process(imageRGB)
   
    height, width, _ = image.shape
    
    landmarks = []
    
    if results.pose_landmarks:
    
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
    return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):

    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def classifyPose(landmarks, output_image, display=False):
    
    label = 'Unknown Pose'

    color = (0, 0, 255)
    
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:                        
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = 'T Pose'

                
    if label != 'Unknown Pose':
        
        color = (0, 255, 0)  
    
    cv2.putText(output_image, label, (10, 20),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    if display:

        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        return output_image, label



def checkHandsJoined(img, draw=False):
    height, width, _ = img.shape

    output_img = img.copy()

    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],left_wrist_landmark[1] - right_wrist_landmark[1]))

    if euclidean_distance < 130:
        hand_status = True
        color = (0, 255, 0)
        
    else:
        hand_status = False
        color = (0, 0, 255)

    if draw:
        pass
        cv2.putText(output_img, f'Hands Joined: {hand_status}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_img, f'Distance: {euclidean_distance}', (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    return output_img, hand_status



if __name__ == '__main__':

    def clickPhoto():
        global takingPhoto
        takingPhoto = True
        print('Saving Image')
        time.sleep(3)
        cv2.imwrite(f'C:\Programming\Python Programming\OpenCV Python\Face Tracking Drone\Images\{time.time()}.jpg', img)
        print('Image Saved')
        takingPhoto = False

    takingPhoto = False
    
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)


    while camera_video.isOpened():
        ok, frame = camera_video.read()    
        frame = cv2.flip(frame, 1)
        img = frame.copy()
        frame_height, frame_width, _ =  frame.shape
        frame = cv2.resize(frame, (1280, 720))
        frame, landmarks = detectPose(frame, pose_video)
        if landmarks:
            frame,hand_status = checkHandsJoined(frame, draw=True)
            frame, label = classifyPose(landmarks, frame, display=False)
            if hand_status and not takingPhoto:
                T = threading.Thread(target=clickPhoto)
                T.start()
        print(takingPhoto)
        if takingPhoto:
            cv2.putText(frame, f'Taking Photo', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
        cv2.imshow('Game', frame)
        k = cv2.waitKey(1) & 0xFF
        if(k == 27):
            break
    camera_video.release()
    cv2.destroyAllWindows()