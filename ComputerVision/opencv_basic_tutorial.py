""" Basic tutorial to read and write Image using OpenCV """
from pathlib import Path
import sys
import getopt
import cv2

def init_image(filename):
    """ Read an image from file """
    return cv2.imread(filename)

def show_image(img, image_name='image'):
    """ Show image from image object """
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.imshow(image_name, img)
    cv2.waitKey(0) #Press q to close the image
    cv2.destroyAllWindows()

def display_image_size(img):
    """ Display image size """
    dim = img.shape
    print(dim)
    print(f'Height: {dim[0]}')
    print(f'Width: {dim[1]}')
    if len(dim) == 3:
        print(f'Channels: {dim[2]}')

def resize_image(img, scale_percent):
    """ Resize image size """
    dim = img.shape
    width = int(dim[1]*scale_percent/100)
    height = int(dim[0]*scale_percent/100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def get_filename_and_extension(filename):
    """ Filename('dog.jpg') => 'dog', 'jpg' """
    path = Path(filename)
    split_parts = path.name.split('.')
    return split_parts[0], split_parts[1]

def get_filename_format(filename, format_name):
    """ Return filename => 'Result\\{filename}_{format_name}.jpg' """
    filename, extension = get_filename_and_extension(filename)
    return f'Results\\{filename}_{format_name}.{extension}'

def get_filename_gray(filename):
    """ Return filename => 'Result\\{filename}_gray.jpg' """
    return get_filename_format(filename, 'gray')

def get_filename_rgb(filename):
    """ Return filename => 'Result\\{filename}_rgb.jpg' """
    return get_filename_format(filename, 'rgb')

def get_filename_from_format(format_value, filename):
    """ Return filename according to format value """
    if format_value == cv2.COLOR_BGR2GRAY:
        return get_filename_gray(filename)
    if format_value == cv2.COLOR_BGR2RGB:
        return get_filename_rgb(filename)
    raise f'{format_value} not supported yet'

def save_image_format(img, filename, format_value):
    """ Save a other-format image for file """
    formatted_image = cv2.cvtColor(img, format_value)
    new_filename = get_filename_from_format(format_value, filename)
    cv2.imwrite(new_filename, formatted_image)
    print('Image saved at '+ new_filename)
    return formatted_image

def save_image_gray(img, filename):
    """ Save a gray-scale image for file """
    return save_image_format(img, filename, cv2.COLOR_BGR2GRAY)

def save_image_rgb(img, filename):
    """ Save a RGB-scale image for file """
    return save_image_format(img, filename, cv2.COLOR_BGR2RGB)

def apply_gaussian_blur(img, filter_size):
    """ Apply Gaussian blur on the image """
    return cv2.GaussianBlur(img, filter_size, 0)

def apply_thresholding(img, threshold=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """ Apply Simple thresholding on image """
    _, thresh = cv2.threshold(img, threshold, max_value, threshold_type)
    return thresh

def apply_adaptive_thresholding(img, adaptive_threshold_type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    """ Apply Adaptive thresholding on the image """
    return cv2.adaptiveThreshold(img, 255, adaptive_threshold_type, cv2.THRESH_BINARY, 11, 2)

def find_contours_in_image(img):
    """ Finds contours in the given image. In OpenCV, finding contours is like finding white object
    from black background. So remember, object to be found should be white and background should be
     black.
    cv2.CHAIN_APPROX_NONE => stores all boundary points for the contour, while
    cv2.CHAIN_APPROX_SIMPLE => stores only required points to draw boundary/contour.
    """
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_contours(img):
    """ Draw contours for edge """
    contours, _ = find_contours_in_image(img)
    print('Printing contours')
    for c_item in contours:
        print(len(c_item))
        cv2.drawContours(img, [c_item], -1, (0, 0, 255), 2)
        show_image(img, image_name='final')

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
    display_image_size(img)
    show_image(img)
    img_gray = save_image_gray(img, filename)
    _ = save_image_rgb(img, filename)

    resized = resize_image(img_gray, 10)
    display_image_size(resized)

    show_image(resized)

    img_blurred = apply_gaussian_blur(resized, (5, 5))
    show_image(img_blurred)

    show_image(apply_thresholding(img_blurred))

    show_image(apply_adaptive_thresholding(img_blurred))
    img_thresholded = apply_adaptive_thresholding(img_blurred,\
        adaptive_threshold_type=cv2.ADAPTIVE_THRESH_MEAN_C)
    show_image(img_thresholded)

    draw_contours(img_thresholded)

    print('End of program')

if __name__ == "__main__":
    main(sys.argv[1:])
