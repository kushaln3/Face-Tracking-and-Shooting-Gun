import numpy as np
import cv2

img = cv2.imread('C:\Programming\Python Programming\OpenCV Python\Tutorial/modi.jpg')
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', imgGray)

imgBlur = cv2.GaussianBlur(imgGray, (19, 19), 0)
cv2.imshow('Blur image', imgBlur)

imgCanny = cv2.Canny(img, 150, 200)
cv2.imshow('Canny image', imgCanny)

imgDialation = cv2.dilate(imgCanny, kernel, iterations=5)
cv2.imshow('Dialated  image', imgDialation)

imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
cv2.imshow('Eroded image', imgEroded)

cv2.waitKey(0)