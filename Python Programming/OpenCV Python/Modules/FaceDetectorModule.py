import mediapipe as mp
import cv2
from time import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # draw = self.draw
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bBoxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                score = round(detection.score[0]*100, 2)
                bBoxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bBox = int(bBoxC.xmin * iw), int(bBoxC.ymin *ih), \
                    int(bBoxC.width * iw), int(bBoxC.height * ih)
                bBoxs.append([id, bBox, score])
                if draw==True:
                    cv2.rectangle(img, bBox, (0,255,0), 2)
                    cv2.putText(img, f"{score}%", (bBox[0], bBox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

        return img, bBoxs

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 100)
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img ,bBoxs = detector.findFaces(img)
        cTime = time()
        fps = int(1//(cTime - pTime))
        pTime = cTime
        cv2.putText(img, f"FPS:{fps}", (0,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        cv2.imshow("Camera", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()