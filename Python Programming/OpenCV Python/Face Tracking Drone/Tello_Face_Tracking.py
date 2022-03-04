import math
import cv2
import cvzone
from utility import *
import KeyPressModule as kp
import time
import threading


# Width and height of the image we are working with
w,h = 640,480

# declaring all the functions that we are going to use
xPLot = cvzone.LivePlot(w=w, yLimit=[-w, w], interval=0.01)
yPLot = cvzone.LivePlot(w=w, yLimit=[-h, h], interval=0.01)
zPLot = cvzone.LivePlot(w=w, yLimit=[0, 15000], interval=0.01)
pdetector = cvzone.PoseDetector(detectionCon=0.75)
kp.init()

def clickPhoto():
    global takingPhoto
    takingPhoto = True
    print('Saving Image')
    time.sleep(3)
    cv2.imwrite(f'C:\Programming\Python Programming\OpenCV Python\Face Tracking Drone\Images\{time.time()}.jpg', originalImage)
    print('Image Saved')
    takingPhoto = False


# PID Values for x,y,z axes
xPid = [0.3,0.0000000001,0.4]
yPid = [0.3,0.0000000001,0.4]

# If debugging with webcam, set it to True
webcam = True
takingPhoto = False
myDrone = None
faceRecognition = False
cap = cv2.VideoCapture(0)
pTime = 0
xpError = 0
ypError = 0
xPID = cvzone.PID(xPid, w // 2,limit=[-100,100])
yPID = cvzone.PID(yPid, h // 2, axis=1, limit=[-100, 100])


# Initialize the tello 
if not webcam:
    myDrone = initializeTello()


while True:

    #Calculating FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    if webcam:
        success, cpimg = cap.read()
        img = cv2.resize(cpimg, (w,h))


        vals, faceRecognition = kp.getKeyboardInput(myDrone,faceRecognition)

        if not faceRecognition:
            print("rc: ", vals)

    else:
        img = telloGetFrame(myDrone, w, h)

        vals, faceRecognition = kp.getKeyboardInput(myDrone,faceRecognition)

        if not faceRecognition:
            myDrone.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    originalImage = img.copy()


# Image processing start
    
    #Find Pose and Landmarks
    img = pdetector.findPose(img, draw=True)
    lmList, bBoxInfo = pdetector.findPosition(img, draw=False)
    
    # UI Elements
    cv2.putText(img, f'FPS: {int(fps)}', (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    if faceRecognition:cv2.putText(img, 'Tracking: {}'.format('ON'), (10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    else:cv2.putText(img, 'Tracking: {}'.format('OFF'), (10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    if not webcam:
        cv2.putText(img, f'Battery: {myDrone.get_battery()}%', (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    

    # Draw centre point of the screen
    cv2.circle(img, (w//2,h//2), 3, (0,0,255), cv2.FILLED)


    if not lmList == []:
        # Coordinates of Nose point
        x = lmList[0][0]
        y = lmList[0][1]
        cords = [[lmList[0][0], lmList[0][1]], 0]
        
        img, label = pdetector.classifyPose(img)

        if label == 'Hands Crossed Pose' and not takingPhoto:
            T = threading.Thread(target=clickPhoto)
            T.start()

        # area = lmList[1]
        
        # Center lines
        cv2.line(img, (w//2,1), (w//2,h), (255,0,0), 1)
        cv2.line(img, (1,h//2), (w,h//2), (255,0,0), 1)


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

        if faceRecognition:
            xpError, ypError = trackFace(myDrone, cords, w, h, xPid, xpError, yPid, ypError)
        if takingPhoto:
            cv2.putText(img, 'Taking Photo', (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))

        stackImage = cvzone.stackImages([img,xImgPlot, yImgPlot], 2, 0.75)
        cv2.resize(stackImage, (1920,1080))
    else:
        if webcam:
            print('rc: [0,0,0,0]')
        else:
            myDrone.send_rc_control(0,0,0,0)
        stackImage = cv2.putText(img, 'No Face Detected', (10,h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))



# image processing end
    cv2.imshow('Game', stackImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if not webcam:
            myDrone.land()
        break


    cv2.waitKey(1)
