import cv2
import urllib.request
import numpy as np
import imutils
from skimage import io

def __url_to_image__(url):
    ''' Steps:
    1. Download the image
    2. Convert it to NumPy array
    3. Read array to openCV format
    '''
    print ('Using urllib and numpy')
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def __url_to_image_scikit__(url):
    ''' Using skimage.io to read and initialize OpenCV image from url'''
    print ('Using skimage.io')
    image = io.imread(url)
    return image

def get_image_from_url(url, method = None):
    ''' Get OpenCV image directly from url. '''
    if method is None:
        method = __url_to_image_scikit__

    return method(url)


if __name__ == "__main__":
    url = 'http://answers.opencv.org/upfiles/logo_2.png'
    img = get_image_from_url(url)
    resized = imutils.resize(img, height=500)
    cv2.imshow('image', img)
    cv2.imshow('resized', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = get_image_from_url(url)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    img = get_image_from_url(url, method=__url_to_image__)
    cv2.imshow('image', img)
    cv2.waitKey(0)
