import cv2
import mediapipe as mp
import time
import numpy as np
 
class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands() # <---- Change Hands Detection parameters here
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.gestureList = ['Index', 'Victory', 'Threeshul', 'Four', 'Palm', 'Spoidermon', 'Super', 'Pinky', 'SideThumb', 'Fist', 'Gun', 'L']

    def rightHand(self, draw=False):

        hands_status = {'Right': False, 'Left': False, 'Right_index' : None, 'Left_index': None}
        for hand_index, hand_info in enumerate(self.results.multi_handedness):
            hand_type = hand_info.classification[0].label
            hands_status[hand_type] = True
            hands_status[hand_type + '_index'] = hand_index 
        if draw: 
            cv2.putText(self.img, hand_type + ' Hand Detected', (10, (hand_index+1) * 30),cv2.FONT_HERSHEY_PLAIN,
                        2, (0,255,0), 2)
        self.hands_status = hands_status
        return hands_status



    def findHands(self, img, draw=True):
        self.img = img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        img.flags.writeable = False
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img, self.results
 
    def findPosition(self, img, handNo=0, draw=True):
 
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
 
        return self.lmList

    def upsideHand(self,lmList):
        # upside Down hand detection detection
        if lmList[5][2] > lmList[0][2]:
            upsideHand = True
        else:
            upsideHand = False
        return upsideHand


    def fingersOpen(self):
        fingers = []
        lmList = self.lmList
        # Thumb
        if self.hands_status['Right']:

            if self.upsideHand(lmList):       
                if lmList[2][1] < lmList[0][1]:
                    rightHand = False
                    if lmList[3][1] > lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    rightHand = True
                    if lmList[3][1] < lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            else:
                if lmList[2][1] > lmList[0][1]:
                    rightHand = False
                    if lmList[3][1] < lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    rightHand = True
                    if lmList[3][1] > lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

        else:
            
            if self.upsideHand(lmList):      
                if lmList[2][1] < lmList[0][1]:
                    rightHand = False
                    if lmList[3][1] > lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    rightHand = True
                    if lmList[3][1] < lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            else:
                if lmList[2][1] > lmList[0][1]:
                    rightHand = False
                    if lmList[3][1] < lmList[4][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    rightHand = True
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




    def drawBoundingBoxes(self, image, results, hand_status, padd_amount = 10, draw=True):
        '''
        This function draws bounding boxes around the hands and write their classified types near them.
        Args:
            image:       The image of the hands on which the bounding boxes around the hands needs to be drawn and the 
                        classified hands types labels needs to be written.
            results:     The output of the hands landmarks detection performed on the image on which the bounding boxes needs
                        to be drawn.
            hand_status: The dictionary containing the classification info of both hands. 
            padd_amount: The value that specifies the space inside the bounding box between the hand and the box's borders.
            draw:        A boolean value that is if set to true the function draws bounding boxes and write their classified 
                        types on the output image. 
            display:     A boolean value that is if set to true the function displays the output image and returns nothing.
        Returns:
            output_image:     The image of the hands with the bounding boxes drawn and hands classified types written if it 
                            was specified.
            output_landmarks: The dictionary that stores both (left and right) hands landmarks as different elements.
        '''
        
        # Create a copy of the input image to draw bounding boxes on and write hands types labels.
        output_image = image.copy()
        
        # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
        output_landmarks = {}

        # Get the height and width of the input image.
        height, width, _ = image.shape

        # Iterate over the found hands.
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Initialize a list to store the detected landmarks of the hand.
            landmarks = []

            # Iterate over the detected landmarks of the hand.
            for landmark in hand_landmarks.landmark:

                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))

            # Get all the x-coordinate values from the found landmarks of the hand.
            x_coordinates = np.array(landmarks)[:,0]
            
            # Get all the y-coordinate values from the found landmarks of the hand.
            y_coordinates = np.array(landmarks)[:,1]
            
            # Get the bounding box coordinates for the hand with the specified padding.
            x1  = int(np.min(x_coordinates) - padd_amount)
            y1  = int(np.min(y_coordinates) - padd_amount)
            x2  = int(np.max(x_coordinates) + padd_amount)
            y2  = int(np.max(y_coordinates) + padd_amount)

            # Initialize a variable to store the label of the hand.
            label = "Unknown"
            
            # Check if the hand we are iterating upon is the right one.
            if hand_status['Right_index'] == hand_index:
                
                # Update the label and store the landmarks of the hand in the dictionary. 
                label = 'Right Hand'
                output_landmarks['Right'] = landmarks
            
            # Check if the hand we are iterating upon is the left one.
            elif hand_status['Left_index'] == hand_index:
                
                # Update the label and store the landmarks of the hand in the dictionary. 
                label = 'Left Hand'
                output_landmarks['Left'] = landmarks
            
            # Check if the bounding box and the classified label is specified to be written.
            if draw:
                
                # Draw the bounding box around the hand on the output image.
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
                
                # Write the classified label of the hand below the bounding box drawn. 
                cv2.putText(output_image, label, (x1, y2+25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20,255,155), 1, cv2.LINE_AA)
        
        return output_image, output_landmarks
    

    def findGesture(self):
        fingers = self.fingersOpen()
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

 
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture("C:\Programming\Python Programming\OpenCV Python\Modules\Resources\Public_Speaking.mp4")
    # cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 100)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img ,results = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            hands_status = detector.rightHand()
            img, landmarks = detector.drawBoundingBoxes(img, results, hands_status)
            gesture = detector.findGesture()
            cv2.putText(img, str(gesture[1]), (30,50), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255))
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
 
if __name__ == "__main__":
    main()