import cv2
import numpy as np
import matplotlib.pyplot as plt
import anisotropic_diffusion_filter as adf

def trrrrrr(img):
    gauss_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    return cv2.add(img, gauss_noise)

def main():
    img = cv2.imread('image.png')
    img_grayscale_basic = img[ : , : ,0] / 3 + img[ : , : ,1] / 3 + img[ : , : ,2] / 3
    img_grayscale_basic = np.array(img_grayscale_basic, dtype=np.uint8)
    noise_img = trrrrrr(img_grayscale_basic)

    cv2.imshow('Grayscale Image', img_grayscale_basic)
    cv2.imshow('Noise Image', noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    adf.adf_convolution(noise_img, 3, 3)
main()