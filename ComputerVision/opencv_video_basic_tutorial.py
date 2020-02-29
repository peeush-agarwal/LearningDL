import cv2

filename = r'Data/People.mp4'

cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()