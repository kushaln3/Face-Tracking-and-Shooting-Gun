from turtle import left
import HandGestureModule as hgm
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui

detector = hgm.htm.handDetector()
gdetector = hgm.handGestures()
# Capture Webcam feed

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 100)
pTime = 0

rightkeypressed = False
leftkeypressed = False
enterkeypressed = False
gamestarted = False

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        gesture = gdetector.findGesture(lmList)
        
        if gamestarted:

            if gesture[1] == 'Palm' and not rightkeypressed:
                print('back')
                pyautogui.keyUp('right')
                pyautogui.keyDown('left')
            
            elif gesture[1] == 'Fist' and not leftkeypressed:
                print('front')
                pyautogui.keyUp('left')
                pyautogui.keyDown('right')

            elif gesture[1] == 'SideThumb':
                print('Nothing')
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')
            
            elif gesture[1] == 'Spoidermon':
                pyautogui.press('return')

        else:
            cv2.putText(img,'Do Spoidermon Pose to start game',(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            if gesture[1] == 'Spoidermon':
                gamestarted = True






    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Camera",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break