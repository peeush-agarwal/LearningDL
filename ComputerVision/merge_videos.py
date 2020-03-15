import cv2
import imutils

video_files = [r'Data/Nature_clip2.mp4', r'Data/People.mp4']

frame_width = 640
frame_height = 360
out_fps = 25
out = cv2.VideoWriter(r"Data/Nature_merge.avi", cv2.VideoWriter.fourcc('M','J','P','G'), out_fps, (frame_width,frame_height), 1)

for idx, file in enumerate(video_files):
    print(f'Processing {file}')
    cap = cv2.VideoCapture(file)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
            
        if ret:
            resized = imutils.resize(frame, frame_width, frame_height)
            cv2.imshow('frame', resized)
            out.write(resized)
            
            cv2.waitKey(1)
        else:
            break

    cap.release()
out.release()
cv2.destroyAllWindows()