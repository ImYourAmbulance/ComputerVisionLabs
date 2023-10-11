import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

IMAGE_REL_PATH = "../../res/house.jpg"
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
    last_frame = np.array([])
    while(cap.isOpened()):
        ret, frame = cap.read()
        if last_frame.size != 0:
            cv2.imshow('frame', frame_modifier_cb(frame - last_frame))

        last_frame = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_image(image_path):
    return cv2.imread(image_path)


def gray_world(img): 
    r_avg, g_avg, b_avg = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    rgb_avg = (r_avg + g_avg + b_avg) / 3
    
    r_coeff = r_avg / rgb_avg
    g_coeff = g_avg / rgb_avg
    b_coeff = b_avg / rgb_avg

    np.multiply(img[:, :, 0], r_coeff, out=img[:, :, 0], casting='unsafe')
    np.multiply(img[:, :, 1], g_coeff, out=img[:, :, 1], casting='unsafe')
    np.multiply(img[:, :, 2], b_coeff, out=img[:, :, 2], casting='unsafe')


if __name__ == "__main__":
    image = load_image(get_abs_path(IMAGE_REL_PATH))

    show_image(image)
    
    gray_world(image)

    show_image(image)
