import cv2
import numpy as np
import FaceDetectorModule as ftm
from cvzone import PID
import time


detector = ftm.FaceDetector(0.5)



def findFaces(img):
    img, bBoxs = detector.findFaces(img)
    myFaceListC = []
    myFaceListArea = []
    for bBox in bBoxs:
        for id in bBox:
            x,y,w,h = bBox[1][0],bBox[1][1],bBox[1][2],bBox[1][3]
            cx = x+w//2
            cy = y+h//2
            area = w*h
            myFaceListArea.append(area)
            myFaceListC.append([cx,cy])
            # cv2.circle(img, (cx,cy), 5, cv2.FILLED)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        cords = [myFaceListC[i], myFaceListArea[i]]
    else: cords = [[0,0],0]
    return img,cords

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def trackFace(cords, w, h, xPid, xpError, yPid, ypError):

    ## PID
    x = cords[0][0]
    y = cords[0][1]

    xError = x - w//2
    
    xSpeed = xPid[0]*xError + xPid[2]*(xError-xpError)
    xSpeed = translate(xSpeed, 320, -320, 0, 180)
    xSpeed = int(np.clip(xSpeed, 0, 180))

    yError = y - h//2
    
    ySpeed = yPid[0]*yError + yPid[2]*(yError-ypError)
    ySpeed = translate(ySpeed, -240, 240, 0, 180)

    ySpeed = int(np.clip(ySpeed, 0, 180))
    print(f'rc: 0,0,{ySpeed},{xSpeed}')
 

    return xError, yError, xSpeed, ySpeed



