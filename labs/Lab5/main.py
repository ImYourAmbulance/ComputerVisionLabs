import cv2  
import numpy as np
import os

IMAGE_REL_PATH = "../../../bin/res/images/skeleton.jpg"
MASK_LAPL = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
MASK_SOBEL = np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]])
MASK_AVG25 = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                [1/25, 1/25, 1/25, 1/25, 1/25],
                [1/25, 1/25, 1/25, 1/25, 1/25],
                [1/25, 1/25, 1/25, 1/25, 1/25],
                [1/25, 1/25, 1/25, 1/25, 1/25]])


def get_abs_path(rel_path):
    return os.path.abspath(os.path.join(__file__ , rel_path))


def load_image(image_path, to_grayscale = False):
    if (to_grayscale):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path)


def show_image(img, title = 'Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gamma_correction(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def filter_with_mask(image, mask): 
    return cv2.filter2D(image, -1, mask)


if __name__ == "__main__":
    base_image = load_image(get_abs_path(IMAGE_REL_PATH), True)
    show_image(base_image)

    lapl_image = filter_with_mask(base_image, MASK_LAPL)
    show_image(lapl_image)

    combined_image = cv2.bitwise_or(base_image, lapl_image)
    show_image(combined_image)

    sobel_image = filter_with_mask(combined_image, MASK_SOBEL)
    show_image(sobel_image)

    e = filter_with_mask(sobel_image, MASK_AVG25)
    show_image(e)

    f = cv2.bitwise_and(sobel_image, e)
    g = cv2.bitwise_or(base_image, f)
    h = gamma_correction(g, 0.5)
    show_image(h)


