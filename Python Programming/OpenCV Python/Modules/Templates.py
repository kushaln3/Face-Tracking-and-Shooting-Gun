#Imports


import cv2
import mediapipe as mp
import time
import math
import numpy as np


# Capture Webcam feed

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 100)
pTime = 0


while True:
    success, img = cap.read()




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Camera",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break