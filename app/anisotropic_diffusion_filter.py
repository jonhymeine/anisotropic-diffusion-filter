import math
import numpy as np

def calculate_decay_factor(lambida, tau, alpha, beta):
    diff = abs(alpha - beta)
    inner_exp = math.exp(-(diff ** (1 / 5) / lambida) / 5)
    decay_factor = (1 - math.exp(-8 * tau * inner_exp)) / 8
    return decay_factor

def get_decay_factor_lut(lambida, tau):
    decay_factor_lut = []
    for i in range(256):
        decay_factor_lut.append(calculate_decay_factor(lambida, tau, 255, 255 - i))
    return decay_factor_lut

def build_kernel(img_area, decay_factor_lut):
    kernel = np.zeros((3, 3), dtype=float)
    alpha = img_area[1, 1]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            value_index = int(alpha - img_area[i, j].astype(float))
            kernel[i, j] = decay_factor_lut[value_index]

    kernel[1, 1] = 1 - np.sum(kernel)
    return kernel

def apply_convolution(img, lambida, tau, iterations):
    decay_factor_lut = get_decay_factor_lut(lambida, tau)
    img_height, img_width = img.shape

    output = img.copy()
    for i in range(iterations):
        padded_img = np.pad(output, pad_width=1, mode='edge')

        for i_img in range(1, img_height + 1):
            for j_img in range(1, img_width + 1):
                img_area = padded_img[i_img-1:i_img+2, j_img-1:j_img+2]
                kernel = build_kernel(img_area, decay_factor_lut)

                output[i_img-1, j_img-1] = np.sum(img_area * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)
