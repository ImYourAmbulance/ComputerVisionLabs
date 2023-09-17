import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

IMAGE_REL_PATH = "../../res/dark.jpg"
VIDEO_REL_PATH = "../../res/dark_video.mp4"

def get_abs_path(rel_path):
    return os.path.abspath(os.path.join(__file__ , rel_path))


def show_image(img):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_grayscale_pixel(pixel):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]
    return (r + g + b) / 3


def to_grayscale_image_average(img): 
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            gray_pixel = to_grayscale_pixel(img[x][y])
            img[x][y] = gray_pixel
    return img


def to_grayscale_image(img, r_factor, g_factor, b_factor): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #return cv2.multiply(img, (r_factor, g_factor, b_factor, 0))


def get_img_diff(img_1, img_2):
    return img_1

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', to_grayscale_image(frame, r_factor = 0.3, g_factor = 0.59, b_factor = 0.11))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_image(image_path):
    return cv2.imread(image_path)

if __name__ == "__main__":
    image = load_image(get_abs_path(IMAGE_REL_PATH))
    
    image = to_grayscale_image(image, r_factor = 0.3, g_factor = 0.59, b_factor = 0.11)
    
    
    play_video(get_abs_path(VIDEO_REL_PATH))

