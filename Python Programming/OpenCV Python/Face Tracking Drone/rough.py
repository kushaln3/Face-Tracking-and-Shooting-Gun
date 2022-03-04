"""
Pose Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""
import cv2
import mediapipe as mp
import math


class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpPose = mp.solutions.mediapipe.python.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.img = img
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][0] - self.lmList[11][0]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][0] - ad
                x2 = self.lmList[15][0] + ad
            else:
                x1 = self.lmList[12][0] - ad
                x2 = self.lmList[11][0] + ad

            y2 = self.lmList[29][1] + ad
            y1 = self.lmList[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    def findAngle(self, p1, p2, p3, draw=True):
        """
        Finds angle between three points. Inputs index values of landmarks
        instead of the actual points.
        :param img: Image to draw output on.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param p3: Point3 - Index of Landmark 3.
        :param draw:  Flag to draw the output on the image.
        :return:
        """

        # Get the landmarks
        x1, y1, _ = p1
        x2, y2, _ = p2
        x3, y3, _ = p3

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(self.img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(self.img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(self.img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(self.img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(self.img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(self.img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][0:]
        x2, y2 = self.lmList[p2][0:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def angleCheck(self, myAngle, targetAngle, addOn=20):
        return targetAngle - addOn < myAngle < targetAngle + addOn



    def classifyPose(self, output_image):
        
        label = 'Unknown Pose'

        color = (0, 0, 255)
        
        left_elbow_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                        self.lmList[self.mpPose.PoseLandmark.LEFT_ELBOW.value],
                                        self.lmList[self.mpPose.PoseLandmark.LEFT_WRIST.value])
        
        right_elbow_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                        self.lmList[self.mpPose.PoseLandmark.RIGHT_ELBOW.value],
                                        self.lmList[self.mpPose.PoseLandmark.RIGHT_WRIST.value])   
        
        left_shoulder_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.LEFT_ELBOW.value],
                                            self.lmList[self.mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                            self.lmList[self.mpPose.PoseLandmark.LEFT_HIP.value])

        right_shoulder_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.RIGHT_HIP.value],
                                            self.lmList[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                            self.lmList[self.mpPose.PoseLandmark.RIGHT_ELBOW.value])

        left_knee_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.LEFT_HIP.value],
                                        self.lmList[self.mpPose.PoseLandmark.LEFT_KNEE.value],
                                        self.lmList[self.mpPose.PoseLandmark.LEFT_ANKLE.value])

        right_knee_angle = self.findAngle(self.lmList[self.mpPose.PoseLandmark.RIGHT_HIP.value],
                                        self.lmList[self.mpPose.PoseLandmark.RIGHT_KNEE.value],
                                        self.lmList[self.mpPose.PoseLandmark.RIGHT_ANKLE.value])
        
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:                        
                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                    label = 'T Pose'
        
                    
        if label != 'Unknown Pose':
            
            color = (0, 255, 0)  
        
        cv2.putText(output_image, label, (10, 20),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        


        return output_image, label


    
    def checkHandsJoined(self, img, draw=False):
        height, width, _ = img.shape

        output_img = img.copy()

        left_wrist_landmark = (self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST].x * width,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST].y * height)
        right_wrist_landmark = (self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST].x * width,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST].y * height)

        euclidean_distance = int(math.hypot(left_wrist_landmark[0] - right_wrist_landmark[0],left_wrist_landmark[1] - right_wrist_landmark[1]))

        if euclidean_distance < 130:
            hand_status = True
            color = (0, 255, 0)
            
        else:
            hand_status = False
            color = (0, 0, 255)

        if draw:
            pass
            cv2.putText(output_img, f'Hands Joined: {hand_status}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
            cv2.putText(output_img, f'Distance: {euclidean_distance}', (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
            
        return output_img, hand_status




def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
        if bboxInfo:
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
