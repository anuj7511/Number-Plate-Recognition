import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2

FIGSIZE = (5,3)

# change this to plates folder
plate_images = glob('/kaggle/input/anprmosaic21/MosaicPS2/indian_plates/*')

def get_random_plate():
    plate_path = plate_images[np.random.randint(0, 1100)]
    print(plate_path)
    return cv2.imread(plate_path)

def display(img_=None, title=''):
    if img_ is None:
        img_ = get_random_plate()
    elif type(img_)==str:
        img_ = cv2.imread(img_)
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()