from PoseEstimationModule import *
import winsound


pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)


cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
while camera_video.isOpened():
    ok, frame = camera_video.read()    
    # frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ =  frame.shape
    # frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame = cv2.resize(frame, (1280,720))
    frame, landmarks = detectPose(frame, pose_video)
    if landmarks:
        frame, label = classifyPose(landmarks, frame, display=False)
        if label == 'T Pose':
            winsound.Beep(500, 500)
    cv2.imshow('Pose Classification', frame)
    k = cv2.waitKey(1) & 0xFF
    if(k == 27):
        break
camera_video.release()
cv2.destroyAllWindows()