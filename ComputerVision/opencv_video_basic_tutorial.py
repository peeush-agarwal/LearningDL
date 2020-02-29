import cv2
import imutils
import numpy as np 

filename = r'Data/People.mp4'

cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        resized = imutils.resize(frame, width=500)
        ratio = frame.shape[0] / float(resized.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding! 
        #thresh = cv2.threshold(blurred, 70 , 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours in the images
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:

            c = c.astype("float")
            c *= ratio
            c = c.astype("int")

            cv2.drawContours(frame, [c], -1, (0,255,0), 2)
            cv2.imshow("final", frame)
            cv2.waitKey(1)
    else:
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()