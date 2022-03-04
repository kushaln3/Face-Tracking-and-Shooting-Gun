import cv2
import mediapipe as mp
import time
import math
import numpy as np
import HandTrackingModule as htm


class handGestures():
    def __init__(self):
        self.tipIds = [4, 8, 12, 16, 20]
        self.gestureList = ['Index', 'Victory', 'Threeshul', 'Four', 'Palm', 'Spoidermon', 'Super', 'Pinky', 'SideThumb', 'Fist', 'Gun', 'L']

    def rightHand(self,lmList):
        # Right hand or Left hand detection
        if self.upsideHand(lmList):
            if lmList[2][1] < lmList[0][1]:
                rightHand = True
            else:
                rightHand = False
            return rightHand
        else:
            if lmList[2][1] > lmList[0][1]:
                rightHand = True
            else:
                rightHand = False
            return rightHand
    
    
    def upsideHand(self,lmList):
        # upside Down hand detection detection
        if lmList[5][2] > lmList[0][2]:
            upsideHand = True
        else:
            upsideHand = False
        return upsideHand
    
    def fingersOpen(self,lmList):
        fingers = []

        # Thumb
        if self.rightHand(lmList):
            # print('Right Hand')
            if lmList[3][1] < lmList[4][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # print('Left Hand')
            if lmList[3][1] > lmList[4][1]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # 4 Fingers
        if self.upsideHand(lmList):
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] > lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers

        else:
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
            

    def findGesture(self, lmList):
        fingers = self.fingersOpen(lmList)
        gL = self.gestureList
        
        # Index
        if fingers == [0,1,0,0,0]:
            gesture = gL[0]
            gestureID = 0
        
        # Victory
        elif fingers == [0,1,1,0,0]:
            gesture = gL[1]
            gestureID = 1
        
        # Threeshul
        elif fingers == [0,1,1,1,0]:
            gesture = gL[2]
            gestureID = 2

        # Four
        elif fingers == [0,1,1,1,1]:
            gesture = gL[3]
            gestureID = 3
            
        # Palm
        elif fingers == [1,1,1,1,1]:
            gesture = gL[4]
            gestureID = 4
        
        # Spoidermon
        elif fingers == [1,1,0,0,1]:
            gesture = gL[5]
            gestureID = 5
        
        # Super
        elif fingers == [0,0,1,1,1]:
            gesture = gL[6]
            gestureID = 6
            
        # Pinky
        elif fingers == [0,0,0,0,1]:
            gesture = gL[7]
            gestureID = 7
            
        # SideThumb
        elif fingers == [1,0,0,0,0]:
            gesture = gL[8]
            gestureID = 8
            
        # Fist
        elif fingers == [0,0,0,0,0]:
            gesture = gL[9]
            gestureID = 9
            
        # Gun
        elif fingers == [1,1,1,0,0]:
            gesture = gL[10]
            gestureID = 10
            
        # L
        elif fingers == [1,1,0,0,0]:
            gesture = gL[11]
            gestureID = 11

        # Unknown
        else:
            gesture = 'Unknown'
            gestureID = 12
        
        return (gestureID, gesture, fingers)



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    pTime = 0
    detector = htm.handDetector()
    mpDraw = mp.solutions.drawing_utils
    gdetector = handGestures()


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