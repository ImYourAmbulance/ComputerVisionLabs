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
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_grayscale_image_average(img): 
    average_values = np.mean(img, axis=2, keepdims=True)
    return np.tile(average_values, (1, 1, 3)).astype(np.uint8)

def to_grayscale_image(img, r_factor = 0.3, g_factor = 0.59, b_factor = 0.11): 
    height, width, n_channels = img.shape
    r_matr = np.full((height, width), r_factor)
    g_matr = np.full((height, width), g_factor)  
    b_matr = np.full((height, width), b_factor)

    sum = np.array(img[:, :, 0] * r_matr + img[:, :, 1] * g_matr + img[:, :, 2] * b_matr).reshape((height, width, 1))
    return np.tile(sum, (1, 1, n_channels)).astype(np.uint8)
    

def play_video(video_path, frame_modifier_cb):
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame_modifier_cb(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_image(image_path):
    return cv2.imread(image_path)

if __name__ == "__main__":
    image = load_image(get_abs_path(IMAGE_REL_PATH))

    image_1 = to_grayscale_image_average(image)
    image_2 = to_grayscale_image(image)
    show_image(image_1 - image_2)
    #show_image(image_2)
    #show_image(image_1 - image_2)
    
    #play_video(get_abs_path(VIDEO_REL_PATH), to_grayscale_image)

