import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import sys

IMAGE_REL_PATH = "../../../bin/res/images/dark.jpg"


def load_image(image_path):
    print(image_path)
    return cv2.imread(image_path)


def get_abs_path(rel_path):
    return os.path.abspath(os.path.join(__file__ , rel_path))


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def support_image_correction(src, dst, output):
    result = src.copy()
    result[:, :, 0] *= src[:, :, 0] / dst[:, :, 0]
    result[:, :, 1] *= src[:, :, 0] / dst[:, :, 0]
    result[:, :, 2] *= src[:, :, 0] / dst[:, :, 0]
    return result


if __name__ == "__main__":# Get the current directory (assuming you are running this script from the project root)
    image = load_image(get_abs_path(IMAGE_REL_PATH))
    show_image(image)
