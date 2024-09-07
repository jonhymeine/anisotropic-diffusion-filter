import math
import cv2
import numpy as np

def crazy_calc(lambida, tau, alpha, beta):
    diff = alpha - beta
    if diff < 0:
        diff *= -1

    inner_exp = math.exp(-(diff ** (1 / 5) / lambida) / 5)
    exp = math.exp(-8 * tau * inner_exp)
    return (1 - exp) / 8

def get_possible_crazy_calc_values(lambida, tau):
    possible_crazy_calc_values = []
    for i in range(256):
        possible_crazy_calc_values.append(crazy_calc(lambida, tau, 255, 255 - i))
    return possible_crazy_calc_values

def build_kernel(img_area, possible_crazy_calc_values):
    k_height, k_width = img_area.shape

    height_center = k_height // 2
    width_center = k_width // 2
    
    kernel = []
    values_sum = 0
    for i in range(k_height):
        for j in range(k_width):
            if i == height_center and j == width_center:
                continue
            value = possible_crazy_calc_values[int(img_area[i, j])]
            values_sum += value
            kernel[i, j] = value
    
    kernel[height_center, width_center] = 1 - values_sum
    return kernel

def add_padding(img, padding_height, padding_width):
    img_height, img_width = img.shape

    padded_img = np.zeros((img_height + padding_height * 2, img_width + padding_width * 2))
    padded_img[padding_height : img_height + padding_height, padding_width : img_width + padding_width] = img

    return padded_img

def adf_convolution(img, k_width, k_height, lambida, tau, has_padding=True):
    possible_crazy_calc_values = get_possible_crazy_calc_values(lambida, tau)
    img_height, img_width = img.shape

    pad_height = k_height // 2
    pad_width = k_width // 2

    if has_padding:
        padded_img = add_padding(img, pad_height, pad_width)

    output = np.zeros((img_height, img_width), dtype=float)

    for i_img in range(img_height):
        for j_img in range(img_width):
            img_area = padded_img[i_img : i_img + pad_height + 1, j_img : j_img + pad_width + 1]
            kernel = build_kernel(img_area, possible_crazy_calc_values)
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    # continuar aqui
                    output[i_img, j_img] += padded_img[i_img + i_kernel, j_img + j_kernel] * kernel[i_kernel, j_kernel]
            output[i_img, j_img] = int(output[i_img, j_img])

    return np.array(output, dtype=np.float32)