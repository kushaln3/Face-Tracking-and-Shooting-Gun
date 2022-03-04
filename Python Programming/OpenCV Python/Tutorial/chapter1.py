import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 100)

while True:
    success, img = cap.read()
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('video' ,img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    