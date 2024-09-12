import math
import cv2
import numpy as np

# calculate_exponential_diff
# adjusted_exp_decay
# calculate_decay_factor
# exponential_decay_by_difference
# use_a_labida_pra_calcular


def crazy_calc(lambida, tau, alpha, beta):
    diff = alpha - beta
    if diff < 0:
        diff *= -1

    inner_exp = math.exp(-(diff ** (1 / 5) / lambida) / 5)
    weight = (1 - math.exp(-8 * tau * inner_exp)) / 8
    return weight

# generate_decay_value_series
# calculate_value_range_by_difference
# compute_decay_series_by_steps
# use_a_tau_da_lambida

def get_possible_crazy_calc_values(lambida, tau):
    possible_crazy_calc_values = []
    for i in range(256):
        possible_crazy_calc_values.append(crazy_calc(lambida, tau, 255, 255 - i))
    return possible_crazy_calc_values

#build_kernel
#build_weighted_kernel
#create_convolution_kernel
#bob_the_builder

def build_kernel(img_area, possible_crazy_calc_values):

    kernel = np.zeros((3, 3), dtype=float)
    alpha = img_area[1, 1]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            value_index = int(alpha - img_area[i, j].astype(float))
            kernel[i, j] = possible_crazy_calc_values[value_index]

    kernel[1, 1] = 1 - np.sum(kernel)
    return kernel

#apply_adf_convolution
#perform_adf_convolution
#apply_adf_filter
#ai_to_tendo_convulsao_ai

def adf_convolution(img, lambida, tau, iterations):
    possible_crazy_calc_values = get_possible_crazy_calc_values(lambida, tau)
    img_height, img_width = img.shape

    output = img.copy()
    for i in range(iterations):
        padded_img = np.pad(output, pad_width=1, mode='constant', constant_values=0)

        for i_img in range(1, img_height + 1):
            for j_img in range(1, img_width + 1):
                img_area = padded_img[i_img-1:i_img+2, j_img-1:j_img+2]
                kernel = build_kernel(img_area, possible_crazy_calc_values)

                output[i_img-1, j_img-1] = np.sum(img_area * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

def adf_convolution2(img, lambda_, tau, iterations):
    img_height, img_width = img.shape
    output_img = img.copy()

    # Realiza a difusão anisotrópica por iterações
    for it in range(iterations):
        padded_img = np.pad(output_img, pad_width=1, mode='constant', constant_values=0)
        
        for i in range(1, img_height + 1):
            for j in range(1, img_width + 1):
                # Extrai a janela 3x3 ao redor do pixel central
                M = padded_img[i-1:i+2, j-1:j+2]
                P = padded_img[i, j]  # pixel central
                
                # Calcula os pesos da janela
                W = np.zeros((3, 3))
                for m in range(3):
                    for n in range(3):
                        if m == 1 and n == 1:  # Ponto central
                            continue
                        W[m, n] = crazy_calc(lambda_, tau, P, M[m, n])
                
                # Normaliza o peso central
                wC = 1 - np.sum(W)
                W[1, 1] = wC
                
                # Aplica a convolução
                new_value = np.sum(M * W)
                output_img[i-1, j-1] = new_value
        
    return np.clip(output_img, 0, 255).astype(np.uint8)