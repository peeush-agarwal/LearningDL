import cv2
import urllib.request
import numpy as np
import imutils

def url_to_image(url):
    ''' Steps:
    1. Download the image
    2. Convert it to NumPy array
    3. Read array to openCV format
    '''
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

if __name__ == "__main__":
    url = 'http://answers.opencv.org/upfiles/logo_2.png'
    img = url_to_image(url)
    resized = imutils.resize(img, height=500)
    cv2.imshow('image', img)
    cv2.imshow('resized', resized)
    cv2.waitKey(0)
