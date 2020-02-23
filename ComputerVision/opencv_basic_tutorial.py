import cv2
import os
import sys, getopt

def init_image(filename):
    return cv2.imread(filename)

def show_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0) #Press q to close the image
    cv2.destroyAllWindows()

def save_image_gray(filename):
    splitParts = os.path.splitext(filename)
    filename_withoutExt = splitParts[0]
    extension = splitParts[1]
    gray_image = cv2.imread(filename, 0)
    new_filename = f'{filename_withoutExt}_gray{extension}'
    cv2.imwrite(new_filename, gray_image)
    print ('Gray image saved at '+ new_filename)

# cv2.IMREAD_COLOR(1) : Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_GRAYSCALE(0) : Loads image in grayscale mode
# cv2.IMREAD_UNCHANGED(-1) : Loads image as such including alpha channel

# Note: OpenCV reads the image in BGR format


def main(argv):
    filename = 'Data\dog.jpg'
    try:
        opts, args = getopt.getopt(argv,"hf:",["filename="])
    except getopt.GetoptError:
        print ('open_cv_tutorial.py -f <filename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('open_cv_tutorial.py -f <filename>\nProvide image filename')
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
    print ('Image filename :', filename)

    img = init_image(filename)
    show_image(img)
    save_image_gray(filename)
    
    print ('End of program')

if __name__ == "__main__":
   main(sys.argv[1:])