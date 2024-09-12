import cv2
import numpy as np

def add_gauss_noise(img):
    gauss_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    return cv2.add(img, gauss_noise)

def add_salt_pepper_noise(img, salt_prob, pepper_prob):
    salt_pepper_noise = np.copy(img)
    num_salt = np.ceil(salt_prob * img.size)
    num_pepper = np.ceil(pepper_prob * img.size)

    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    salt_pepper_noise[coords_salt[0], coords_salt[1]] = 255

    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    salt_pepper_noise[coords_pepper[0], coords_pepper[1]] = 0

    return salt_pepper_noise

def add_poisson_noise(img):
    img_poisson_noise = np.copy(img)
    noise = np.random.poisson(img_poisson_noise.astype(np.float32) / 255.0 * 40.0) / 40.0 * 255.0
    img_poisson_noise = img_poisson_noise + noise
    img_poisson_noise = np.clip(img_poisson_noise, 0, 255).astype(np.uint8)
    return img_poisson_noise

def add_speckle_noise(img):
    speckle_noise = np.random.randn(img.shape[0], img.shape[1]) * 0.2
    img_speckle_noise = img + img * speckle_noise
    img_speckle_noise = np.clip(img_speckle_noise, 0, 255).astype(np.uint8)
    return img_speckle_noise