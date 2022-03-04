import cv2

img = cv2.imread("C:\Programming\Python Programming\OpenCV Python\Tutorial\modi.jpg")
cv2.imshow('image', img)
# print(img.shape)

imgResize = cv2.resize(img, (640, 480))
cv2.imshow('Resized image', img)

imgCrop = img[0:200, 200:350]
cv2.imshow('Cropped image', imgCrop)

cv2.waitKey(0)