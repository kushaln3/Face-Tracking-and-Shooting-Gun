from djitellopy import Tello
import cv2
import numpy as np
import FaceDetectorModule as ftm
from cvzone import PID
import time


detector = ftm.FaceDetector(0.5)


def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.xSpeed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone


def telloGetFrame(myDrone, w=360, h=240):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

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


def trackFace(myDrone, cords, w, h, xPid, xpError, yPid, ypError):

    ## PID
    x = cords[0][0]
    y = cords[0][1]
    area = cords[1]

    xError = x - w//2
    
    xSpeed = xPid[0]*xError + xPid[2]*(xError-xpError)
    xSpeed = int(np.clip(xSpeed,-100,100))

    yError = y - h//2
    yError = -yError
    
    ySpeed = yPid[0]*yError + yPid[2]*(yError-ypError)
    ySpeed = int(np.clip(ySpeed,-100,100))
    print(f'rc: 0,0,{ySpeed},{xSpeed}')
 
    # print(ySpeed)
    if myDrone != None:
        if x !=0 and y != 0:
            myDrone.yaw_velocity = xSpeed
            myDrone.up_down_velocity = ySpeed
        else:
            print('empty values set')
            myDrone.for_back_velocity = 0
            myDrone.left_right_velocity = 0
            myDrone.up_down_velocity = 0
            myDrone.yaw_velocity = 0
            error = 0
        if myDrone.send_rc_control:
            myDrone.send_rc_control(myDrone.left_right_velocity,
                                    myDrone.for_back_velocity,
                                    myDrone.up_down_velocity,
                                    myDrone.yaw_velocity)
    return xError, yError



def testTrackFace(myDrone, cords, w, h, xPid, xPID, yPid, yPID):

    ## PID
    x = cords[0][0]
    y = cords[0][1]
    area = cords[1]

    
    xSpeed = int(xPID.update(x))
    ySpeed = int(yPID.update(-y))
    
    xError = x - w//2
    yError = x - w//2

    print(f'rc: [0,0,{ySpeed},{xSpeed}]')
 
    # print(ySpeed)
    if myDrone != None:
        if x !=0 and y != 0:
            myDrone.yaw_velocity = xSpeed
            myDrone.up_down_velocity = ySpeed
        else:
            print('empty values set')
            myDrone.for_back_velocity = 0
            myDrone.left_right_velocity = 0
            myDrone.up_down_velocity = 0
            myDrone.yaw_velocity = 0
            error = 0
        if myDrone.send_rc_control:
            myDrone.send_rc_control(myDrone.left_right_velocity,
                                    myDrone.for_back_velocity,
                                    myDrone.up_down_velocity,
                                    myDrone.yaw_velocity)
    return xError, yError


