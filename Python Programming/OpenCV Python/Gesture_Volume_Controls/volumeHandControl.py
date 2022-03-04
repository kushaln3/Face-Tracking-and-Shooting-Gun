from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import winsound
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import numpy as np

detector = htm.handDetector()

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

pTime = 0


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    detector.findHands(img)


    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        length = math.hypot(x1-x2, y1-y2)
        # print(length)
        vol = int(np.interp(length, [50,300], [minVol,maxVol]))
        # print(vol) 
        # volume.SetMasterVolumeLevel(vol, None)

        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

        if length<50:
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            volume.SetMasterVolumeLevel(maxVol, None)
        else:
            volume.SetMasterVolumeLevel(minVol, None)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break