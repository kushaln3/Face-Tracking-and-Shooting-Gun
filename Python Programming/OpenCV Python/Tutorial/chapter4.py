from pickletools import uint8
import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)

cv2.line(img, (0,0), (256,256), (0,255,0), 3)
cv2.line(img, (256,0), (256,256), (0,255,0), 3)
cv2.line(img, (512,0), (256,256), (0,255,0), 3)

cv2.rectangle(img, (256,256), (512,512), (0,0,255), cv2.FILLED)

cv2.putText(img, 'Kushal', (0,256), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0))

cv2.imshow('image', img)
cv2.waitKey(0)