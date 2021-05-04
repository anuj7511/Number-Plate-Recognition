import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import get_random_plate, display


def segment_characters(
    image,
    plate_name,
    clahe_clipLimit=3,
    clahe_tileGridSize=8,
    erosion_iters=1,
    dilation_iters=2,
    display_verbose=1,
):
    """
    Input: image is an BGR image of (better if perspectively transformed) license plate
    Returns: a list of all contours segmented as a character of Plate Number
    """
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (800, 200))
    img_lp = cv2.fastNlMeansDenoisingColored(img_lp, None, 10, 10, 7, 15)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clipLimit, tileGridSize=(clahe_tileGridSize, clahe_tileGridSize)
    )
    img_gray_lp = clahe.apply(img_gray_lp)
    img_binary_lp = cv2.threshold(
        img_gray_lp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    img_binary_lp = cv2.erode(img_binary_lp, (3, 3), iterations=erosion_iters)
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3), iterations=dilation_iters)

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Estimations of character contours sizes of cropped license plates
    dimensions = [20, LP_WIDTH, 40, LP_HEIGHT]

    if display_verbose >= 2:
        """show processed image"""
        plt.imshow(img_binary_lp, cmap=plt.cm.binary)
        plt.show()

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, plate_name, img_binary_lp, display_verbose)
    return char_list


def find_contours(dimensions, plate_name, img, display_verbose=1):
    """
    Filters and returns characters contours from other noise
    """
    # Find all contours in the image
    cntrs, _ = cv2.findContours(
        img.copy(), cv2.RETR_CCOMP + cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Retrieve potential dimensions
    lower_width, upper_width, lower_height, upper_height = dimensions

    # sort contours according to their x position
    coords = []
    for contour in cntrs:
        (x, y, w, h) = cv2.boundingRect(contour)
        coords.append([x, y, w, h])
    coords = sorted(coords, key=lambda x: x[0])

    ii = img.copy()

    x_cntr_list = []
    target_contours = []
    img_res = []
    prev_x = 0

    meanWidth = []
    meanHeight = []
    meanY = []
    for intX, intY, intWidth, intHeight in coords:
        if (
            intWidth > lower_width
            and intWidth < upper_width
            and intHeight > lower_height
            and intHeight < upper_height
            and intX > prev_x
        ):
            meanWidth.append(intWidth)
            meanHeight.append(intHeight)
            meanY.append(intY)
            prev_x = intX + (3 * intWidth) / 4

    meanWidth = sum(meanWidth) / len(meanWidth)
    meanHeight = sum(meanHeight) / len(meanHeight)

    # use the meanY & meanHeight to ignore noisy contours
    lower_Y = sum(meanY) / len(meanY) - meanHeight * 0.50
    upper_Y = sum(meanY) / len(meanY) + meanHeight * 0.75

    prev_x = 0
    plt.figure()
    plt.ion()
    for intX, intY, intWidth, intHeight in coords:
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        # the check `intX>prev_x` filters out child contours of characters from passing the test
        if (
            (lower_width < intWidth < upper_width)
            and (lower_height < intHeight < upper_height)
            and intX > prev_x
            and (lower_Y < intY < upper_Y)
        ):
            prev_x = intX + (3 * intWidth) / 4

            # This check identifies possible merged characters, if true split the contour in 2
            if intWidth > meanWidth * 1.5:
                x_cntr_list.append(
                    intX
                )  # stores the x coordinate of the character's contour, to used later for indexing the contours
                char = img[intY : intY + intHeight, intX : intX + intWidth // 2]

                # converting the contour image `char` into a square by paading with white pixels so characters dont get squashed or stretched on classification
                # helps a lot for classification b/q characters like '1' '7' 'l' 'I' etc
                border_y = (max(char.shape) - char.shape[0]) // 2
                border_x = (max(char.shape) - char.shape[1]) // 2

                char = cv2.copyMakeBorder(
                    char,
                    border_y,
                    border_y,
                    border_x,
                    border_x,
                    cv2.BORDER_CONSTANT,
                    value=255.0,
                )
                char = cv2.resize(char, (24, 24))
                # small padding to center the contours
                char = cv2.copyMakeBorder(
                    char, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255.0
                )

                cv2.rectangle(
                    ii,
                    (intX, intY),
                    (intWidth // 2 + intX, intY + intHeight),
                    (50, 21, 200),
                    2,
                )
                if display_verbose >= 1:
                    """show bounding rectangles for each identified character"""
                    plt.imshow(ii, cmap=plt.cm.binary)

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)
                img_res.append(
                    char
                )  # List that stores the character's binary image (unsorted)

                # repeat process for split 2
                x_cntr_list.append(
                    intX + intWidth // 2
                )  # stores the x coordinate of the character's contour, to used later for indexing the contours
                char = img[
                    intY : intY + intHeight, intX + intWidth // 2 : intX + intWidth
                ]

                border_y = (max(char.shape) - char.shape[0]) // 2
                border_x = (max(char.shape) - char.shape[1]) // 2

                char = cv2.copyMakeBorder(
                    char,
                    border_y,
                    border_y,
                    border_x,
                    border_x,
                    cv2.BORDER_CONSTANT,
                    value=255.0,
                )
                char = cv2.resize(char, (24, 24))
                char = cv2.copyMakeBorder(
                    char, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255.0
                )

                cv2.rectangle(
                    ii,
                    (intX + intWidth // 2, intY),
                    (intWidth + intX, intY + intHeight),
                    (50, 21, 200),
                    2,
                )
                if display_verbose >= 1:
                    """show bounding rectangles for each identified character"""
                    plt.imshow(ii, cmap=plt.cm.binary)

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                img_res.append(
                    char
                )  # List that stores the character's binary image (unsorted)

            else:
                x_cntr_list.append(
                    intX
                )  # stores the x coordinate of the character's contour, to used later for indexing the contours
                # extracting each character using the enclosing rectangle's coordinates.
                char = img[intY : intY + intHeight, intX : intX + intWidth]

                border_y = (max(char.shape) - char.shape[0]) // 2
                border_x = (max(char.shape) - char.shape[1]) // 2

                char = cv2.copyMakeBorder(
                    char,
                    border_y,
                    border_y,
                    border_x,
                    border_x,
                    cv2.BORDER_CONSTANT,
                    value=255.0,
                )
                char = cv2.resize(char, (24, 24))
                char = cv2.copyMakeBorder(
                    char, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255.0
                )

                cv2.rectangle(
                    ii,
                    (intX, intY),
                    (intWidth + intX, intY + intHeight),
                    (50, 21, 200),
                    2,
                )
                if display_verbose >= 1:
                    """show bounding rectangles for each identified character"""
                    plt.imshow(ii, cmap=plt.cm.binary)

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                img_res.append(
                    char
                )  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
    plt.ioff()
    plt.savefig(f"./results/{plate_name.split('/')[-1]}_chars.jpg")
    plt.pause(1)
    plt.close()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(
            img_res[idx]
        )  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res
