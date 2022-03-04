
import pyfirmata
import winsound
from pyfirmata import Arduino, SERVO
import numpy as np
import time
from utility import *
import cv2
import KeyPressModule as kp
import time
import cvzone
import threading
# Width and height of the image we are working with
w,h = 640,480
port = 'COM12'
swPort = 'COM12'

board = Arduino(port)
it = pyfirmata.util.Iterator(board)  
it.start()  

kp.init()

joyxpin = board.get_pin('a:0:i')
joyypin = board.get_pin('a:1:i')


xpin = board.get_pin('d:9:s')
ypin = board.get_pin('d:10:s')
swpin = board.get_pin('d:11:s')
joyswpin = board.get_pin('d:2:i')

xpin.write(90)
ypin.write(90)
swpin.write(90)

# declaring all the functions that we are going to use
xPLot = cvzone.LivePlot(w=w, yLimit=[-w, w], interval=0.01)
yPLot = cvzone.LivePlot(w=w, yLimit=[-h, h], interval=0.01)
zPLot = cvzone.LivePlot(w=w, yLimit=[0, 15000], interval=0.01)
pdetector = cvzone.PoseDetector(detectionCon=0.75)

# PID Values for x,y,z axes
xPid = [0.56,0.0000000001,0.5]
yPid = [0.3,0.0000000001,0.2]
manualSpeed = 0.5
shootMode = True
avgShoot = 0
# If debugging with webcam, set it to True
webcam = True
takingPhoto = False
cap = cv2.VideoCapture(0)
pTime = 0
xpError = 0
ypError = 0
xPID = cvzone.PID(xPid, w // 2,limit=[-100,100])
yPID = cvzone.PID(yPid, h // 2, axis=1, limit=[-100, 100])
mode = 'OFF'

cap = cv2.VideoCapture(2)
pTime = 0


def checkServos():
    xpin.write(180)
    time.sleep(0.2)
    xpin.write(0)
    time.sleep(0.4)
    xpin.write(180)
    time.sleep(0.2)
    xpin.write(90)

    ypin.write(180)
    time.sleep(0.2)
    ypin.write(0)
    time.sleep(0.4)
    ypin.write(180)
    time.sleep(0.2)

    xpin.write(90)
    ypin.write(90)


def shootBullet():
    swpin.write(180)
    time.sleep(1.25)
    swpin.write(0)
    time.sleep(1.25)
    swpin.write(90)



def clickPhoto():
    global takingPhoto
    takingPhoto = True
    print('Saving Image')
    time.sleep(3)
    cv2.imwrite(f'C:\Programming\Python Programming\OpenCV Python\Face Tracking Drone\Images\{time.time()}.jpg', originalImage)
    print('Image Saved')
    takingPhoto = False


try:
    
    while True:
        success, img = cap.read()
        originalImage = img.copy()

    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv2.waitKey(1)
        
    # Image processing start
        
        # UI Elements
        cv2.putText(img, f'FPS: {int(fps)}', (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(img, f'Mode: {mode}', (10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))


        # Center lines
        crosshair_height = 25
        cv2.line(img, (w//2,h//2 - crosshair_height), (w//2,h//2 + crosshair_height), (255,0,0), 1)
        cv2.line(img, (w//2 - crosshair_height,h//2), (w//2 + crosshair_height,h//2), (255,0,0), 1)
           # Draw centre point of the screen
        cv2.circle(img, (w//2,h//2), 3, (0,0,255), cv2.FILLED)

        if mode != 'OFF':
            stackImage = img
        


            if mode == 'TRACKING':
                cv2.putText(img, f'Auto Shoot: {shootMode}', (10,80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
                if kp.getKey('s') and shootMode == False:
                    shootMode = True
                
                if kp.getKey('d') and shootMode == True:
                    shootMode = False
                if kp.getKey('SPACE'):
                    shootBullet()

                #Find Pose and Landmarks
                img = pdetector.findPose(img, draw=True)
                lmList, bBoxInfo = pdetector.findPosition(img, draw=False)
                if not lmList == []:
                    # Coordinates of Nose point
                    x = lmList[0][1]
                    y = lmList[0][2]
                    cords = [[x, y], 0]
                    

                    # area = lmList[1]
                
                    # distance lines
                    cv2.line(img, (w//2, y), (x,y), (255,0,0), 3)
                    cv2.line(img, (x, h//2), (x,y), (255,0,0), 3)

                    cv2.circle(img, (x,y), 3, (0,255,0), cv2.FILLED)
                    
                    xError = x - w//2
                    xImgPlot = xPLot.update(xError)
                    xImgPlot = cv2.resize(xImgPlot, (w,h))

                    yError = y - h//2
                    yImgPlot = yPLot.update(yError)
                    yImgPlot = cv2.resize(yImgPlot, (w,h))

                    # z = area
                    # zImgPlot = zPLot.update(z)
                    # zImgPlot = cv2.resize(zImgPlot, (w,h))
            
                    xpError, ypError, ySpeed, xSpeed = trackFace(cords, w, h, xPid, xpError, yPid, ypError)
                    xpin.write(xSpeed)
                    ypin.write(ySpeed)
                    swpin.write(90)

                    if takingPhoto:
                        cv2.putText(img, 'Taking Photo', (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))

                    stackImage = cvzone.stackImages([img,xImgPlot, yImgPlot], 2, 0.75)
                    cv2.resize(stackImage, (1920,1080))
                    if shootMode:
                        if xError <= 20 and xError >= -20 and yError <= 40 and yError >= -40:
                            avgShoot += 1
                            if avgShoot >= 10:
                                shootBullet()
                                avgShoot = 0
                else:
                    xpin.write(90)
                    ypin.write(90)
                    swpin.write(90)
                    print('rc: [0,0,0,0]')
                    stackImage = cv2.putText(img, 'No Face Detected', (10,h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        
            elif mode == 'MANUAL':
                cv2.putText(img, f'Speed: {int(manualSpeed*100)}', (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
                if kp.getKey('LEFT'):
                    ypin.write(int(translate(manualSpeed, 0, 1, 90, 180)))
                    print(translate(manualSpeed, 0, 1, 90, 180))
                    time.sleep(0.1)

                elif kp.getKey('RIGHT'):
                    ypin.write(int(translate(manualSpeed, 0, 1, 90, 0)))
                    time.sleep(0.1)

                if kp.getKey('UP'):
                    xpin.write(int(translate(manualSpeed, 0, 1, 90, 0)))
                    time.sleep(0.1)

                elif kp.getKey('DOWN'):
                    xpin.write(int(translate(manualSpeed, 0, 1, 90, 180)))
                    time.sleep(0.1)

                if kp.getKey('SPACE'):
                    shootBullet()

                if kp.getKey('EQUALS'):
                    if manualSpeed < 1:
                        manualSpeed += 0.05
                        time.sleep(0.1)
        
                if kp.getKey('MINUS'):
                    if manualSpeed > 0:
                        manualSpeed -= 0.05
                        time.sleep(0.1)
                
                
                if kp.getKey('6'):
                    swpin.write(30)
                    time.sleep(0.1)
                    swpin.write(90)

                elif kp.getKey('4'):
                    swpin.write(150)
                    time.sleep(0.1)
                    swpin.write(90)

                xpin.write(90)
                ypin.write(90)




            # general keyboard commands

            # Quitting: press q
            if kp.getKey('q'):
                print('Exiting')
                xpin.write(90)
                ypin.write(90)
                exit()
            
            if kp.getKey('m'):
                mode = 'MANUAL'

            if kp.getKey('t'):
                mode = 'TRACKING'

            cv2.imshow("Image", stackImage)

        if kp.getKey('LSHIFT'):
            if mode == 'OFF':
                mode = 'ON' 
                winsound.PlaySound('C:\\Users\\HP\\Desktop\\Kushal N Projects\\Automatic gun\\Automatuc_Gun.ino\\startup.wav', winsound.SND_ASYNC)
                # checkServos()
        
        if mode != "OFF":
            if kp.getKey('LCTRL'):
                mode = 'OFF'

        

except KeyboardInterrupt:
    xpin.write(90)
    ypin.write(90)
    swpin.write(90)
    exit()
