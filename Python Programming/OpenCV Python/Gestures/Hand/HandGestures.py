import cv2
import mediapipe as mp
import time
import math
import numpy as np
import HandTrackingModule as htm
import HandGestureModule as hgm



cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
detector = htm.handDetector()
mpDraw = mp.solutions.drawing_utils
gdetector = hgm.handGestures()


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        gesture = gdetector.findGesture(lmList)
        cv2.putText(img, f"{gesture[1]}", (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,0,255))
        cv2.putText(img, f"{gesture[2]}", (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,0,255))




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Camera",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break