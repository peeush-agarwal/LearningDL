import cv2
import imutils
import os

base_path = 'Data/'
input_files = ['Nature_clip2.mp4', 'People.mp4']
output_file = 'Nature_merge.mp4'

frame_width = 640
frame_height = 360
out_fps = 25

def concat_video_files(base_path, input_files, output_file, frame_width, frame_height, out_fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(base_path, output_file), fourcc, out_fps, (frame_width,frame_height))
    # fourcc = cv2.VideoWriter.fourcc('M','J','P','G')
    # out = cv2.VideoWriter(r"Data/Nature_merge.avi", fourcc, out_fps, (frame_width,frame_height), 1)

    for idx, file in enumerate(input_files):
        file_path = os.path.join(base_path, file)
        print(f'Processing {file_path}')
        cap = cv2.VideoCapture(file_path)
        
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

if __name__ == "__main__":
    concat_video_files(base_path, input_files, output_file, frame_width, frame_height, out_fps)
    print('Program completed.')