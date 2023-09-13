import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

IMAGE_REL_PATH = "../../res/dark.jpg"

def get_image_path(rel_path):
    return os.path.abspath(os.path.join(__file__ , rel_path))


def load_image(rel_path):
    image_path = get_image_path(rel_path)
    return cv2.imread(image_path)

if __name__ == "__main__":
    image = load_image(IMAGE_REL_PATH)

    cv2.imshow('Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    