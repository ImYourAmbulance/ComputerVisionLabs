import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

def get_abs_path(rel_path):
    return os.path.abspath(os.path.join(__file__ , rel_path))

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_image(image_path):
    return cv2.imread(image_path)

def test():
    return 54

