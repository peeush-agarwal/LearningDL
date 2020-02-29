import cv2
import imutils
import numpy as np
import imageurl_to_image as url2img

def image_proc_pipeline(img):
    resized = imutils.resize(img, height=500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # edge detection
    edges = cv2.Canny(blurred, 75, 200)
    return resized, edges

def display_image(resized, result):
    cv2.imshow("resized", resized)
    cv2.imshow("result", result)

def display_image_with_original(img, resized, result):
    cv2.imshow("image", img)
    display_image(resized, result)

def process_image_from_url(url):
    img = url2img.get_image_from_url(url)
    resized, result = image_proc_pipeline(img)
    display_image_with_original(img, resized, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_from_file(filename):
    img = cv2.imread(filename)
    resized, result = image_proc_pipeline(img)
    display_image_with_original(img, resized, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_captured_video(cap):
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            resized, result = image_proc_pipeline(frame)

            display_image(resized, result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def process_video_from_webcam():
    cap = cv2.VideoCapture(0)
    process_captured_video(cap)
    
def process_video_from_file(filename):
    cap = cv2.VideoCapture(filename)
    process_captured_video(cap)

if __name__ == '__main__':
    url = 'http://answers.opencv.org/upfiles/logo_2.png'
    process_image_from_url(url)

    filename = r'Data/dog.jpg'
    process_image_from_file(filename)

    filename = r'Data/People.mp4'
    process_video_from_file(filename)

    exit(0)
    