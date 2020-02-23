""" Basic tutorial to read and write Image using OpenCV """
from pathlib import Path
import sys
import getopt
import cv2

def init_image(filename):
    """ Read an image from file """
    return cv2.imread(filename)

def show_image(img):
    """ Show image from image object """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0) #Press q to close the image
    cv2.destroyAllWindows()

def get_filename_and_extension(filename):
    """ Filename('dog.jpg') => 'dog', 'jpg' """
    path = Path(filename)
    split_parts = path.name.split('.')
    return split_parts[0], split_parts[1]

def get_filename_gray(filename):
    """ Return filename => 'Result\{filename}_gray.jpg' """
    filename, extension = get_filename_and_extension(filename)
    return f'Results\\{filename}_gray.{extension}'

def get_filename_rgb(filename):
    """ Return filename => 'Result\{filename}_rgb.jpg' """
    filename, extension = get_filename_and_extension(filename)
    return f'Results\\{filename}_rgb.{extension}'

def save_image_gray(img, filename):
    """ Save a gray-scale image for file """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_filename = get_filename_gray(filename)
    cv2.imwrite(new_filename, gray_image)
    print('Gray image saved at '+ new_filename)

def save_image_rgb(img, filename):
    """ Save a RGB-scale image for file """
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_filename = get_filename_rgb(filename)
    cv2.imwrite(new_filename, rgb_image)
    print('RGB image saved at '+ new_filename)

# cv2.IMREAD_COLOR(1) : Loads a color image. Any transparency of image will be neglected.
#   It is the default flag.
# cv2.IMREAD_GRAYSCALE(0) : Loads image in grayscale mode
# cv2.IMREAD_UNCHANGED(-1) : Loads image as such including alpha channel

def main(argv):
    """ Read, show and save image using openCV """
    filename = r'Data\dog.jpg'
    try:
        opts, _ = getopt.getopt(argv, "hf:", ["filename="])
    except getopt.GetoptError:
        print('open_cv_tutorial.py -f <filename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('open_cv_tutorial.py -f <filename>\nProvide image filename')
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
    print('Image filename :', filename)

    img = init_image(filename)
    show_image(img)
    save_image_gray(img, filename)
    save_image_rgb(img, filename)

    print('End of program')

if __name__ == "__main__":
    main(sys.argv[1:])
