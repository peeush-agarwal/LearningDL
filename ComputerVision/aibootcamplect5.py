import cv2
import imutils
import numpy as np

img = cv2.imread(r'Data/dog.jpg')

def image_proc_pipeline(img):
    resized = imutils.resize(img, height=500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # edge detection
    edges = cv2.Canny(blurred, 75, 200)
    return resized, edges

resized, result = image_proc_pipeline(img)

cv2.imshow("resized", resized)
cv2.imshow("edge detection", result)
# cv2.imshow("cv2 resized", resized_cv2)
cv2.waitKey(0)