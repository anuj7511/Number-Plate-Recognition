import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import get_random_plate, display

# some modified functions used from other sudoku project of Lavish


def show_image(img):
    """function to show an image"""
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


def display_contours(img, contours, color=(0, 255, 0), thickness=5):
    """function to display identified contours of sudoku board"""

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cont_image = cv2.drawContours(img, contours, -1, color, thickness)
    show_image(cont_image)


def display_corners(img, corners, colour=(0, 0, 255), radius=20):
    """function to display corners of sudoku board"""

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for corner in corners:
        img = cv2.circle(img, tuple(corner), radius, colour, -1)
    show_image(img)


def get_contours(img, show_contours):

    # for contour detection, it needs object to be white present in a black background.
    # so, first we will invert the image.
    img = cv2.bitwise_not(img, img)

    # now find contours
    # outer contours(boundry of sudoku)
    ext_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # all contours(numbers, grid lines)
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Now invert the image again after finding contours
    img = cv2.bitwise_not(img, img)

    if show_contours:
        display_contours(img, contours)

    return img, contours


def get_corners(img, contours, show_corners):
    contours = sorted(
        contours, key=cv2.contourArea
    )  # Sorting contours by area in ascending order
    box = contours[-2]

    # A function to obtain the element at 1st position of an element
    # because 1st element will be used as a key to for finding max/min of points below.
    def func(x):
        return x[1]

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in box]), key=func)
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in box]), key=func)
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in box]), key=func)
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in box]), key=func)

    # x, y coordinates of 4 corner points
    bottom_right = box[bottom_right][0]
    top_left = box[top_left][0]
    bottom_left = box[bottom_left][0]
    top_right = box[top_right][0]

    corners = (top_left, top_right, bottom_left, bottom_right)

    if show_corners:
        display_corners(img, corners)

    return corners


def get_cropimage(img, corners):

    top_left, top_right, bottom_left, bottom_right = corners

    def distance_between(p1, p2):
        # Gives the distance between two pixels
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a * 2) + (b * 2))

    input_pts = np.array(
        [top_left + 3, top_right, bottom_left, bottom_right], dtype="float32"
    )

    length = [800, 200]

    output_pts = np.array([[0, 0], [800, 0], [0, 200], [800, 200]], dtype="float32")

    # Gets the transformation matrix for skewing the image to
    # fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Performs the transformation on the original image
    warped = cv2.warpPerspective(img, m, (int(length[0]), int(length[1])))
    return warped


def preprocessForPerspecive(
    img, clahe_clipLimit=7, clahe_tileGridSize=16, erosion_iters=1, dilation_iters=1
):

    # return the resized bgr unprocessed image
    img_original = cv2.resize(img, (800, 200))

    # Preprocess cropped license plate image
    img_lp = cv2.resize(img, (800, 200))
    img_lp = cv2.fastNlMeansDenoisingColored(img_lp, None, 10, 10, 7, 15)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clipLimit, tileGridSize=(clahe_tileGridSize, clahe_tileGridSize)
    )
    img_gray_lp = clahe.apply(img_gray_lp)
    img_binary_lp = cv2.threshold(
        img_gray_lp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3), iterations=erosion_iters)
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3), iterations=dilation_iters)

    BOUDARY_THICKNESS = 10

    img_binary_lp[:BOUDARY_THICKNESS, :] = 0.0
    img_binary_lp[-BOUDARY_THICKNESS:, :] = 0.0
    img_binary_lp[:, :BOUDARY_THICKNESS] = 0.0
    img_binary_lp[:, -BOUDARY_THICKNESS:] = 0.0

    return img_binary_lp, img_original


def perspectiveTransform(
    image,
    clahe_clipLimit=7,
    clahe_tileGridSize=16,
    erosion_iters=1,
    dilation_iters=1,
    show_contours=False,
    show_corners=False,
):
    """
    returns perspective trasformed crop from original image
    shape=(200,800), image format: BGR
    """
    if type(image) == str:
        image = cv2.imread(image)
    processed_img, img_original = preprocessForPerspecive(
        image,
        clahe_clipLimit=clahe_clipLimit,
        erosion_iters=erosion_iters,
        dilation_iters=dilation_iters,
    )

    img, ext_contours = get_contours(processed_img, show_contours)
    corners = get_corners(img, ext_contours, show_corners)
    final_plate = get_cropimage(img_original, corners)

    return final_plate


if __name__ == "__main__":
    image = get_random_plate()
    display(image)

    image = perspectiveTransform(
        image,
        clahe_clipLimit=3,
        clahe_tileGridSize=32,
        erosion_iters=0,
        dilation_iters=5,
        show_contours=True,
        show_corners=True,
    )
    display(image)
