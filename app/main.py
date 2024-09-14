import cv2
import numpy as np
import anisotropic_diffusion_filter as adf
import noise
import quality_metrics as qm

def apply_sobel(img):
    img = img.astype(np.float32)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return np.clip(sobel_combined, 0, 255).astype(np.uint8)

def apply_UMHF(img, k):
    img = img.astype(np.float32)
    gaussian_blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    img_mask = img - gaussian_blur_img
    return np.clip((img + (k * img_mask)), 0 , 255).astype(np.uint8)

# 1
gs_img = cv2.imread('images/image.png', 0)

# 2
# noise_img = noise.add_gauss_noise(gs_img)
noise_img = noise.add_salt_pepper_noise(gs_img, 0.05, 0.05)
# noise_img = noise.add_poisson_noise(gs_img)
# noise_img = noise.add_speckle_noise(gs_img)

# 3
adf_img = adf.apply_convolution(noise_img, 0.7, 0.3, 5)
gauss_img = cv2.GaussianBlur(noise_img, (3, 3), 0)

cv2.imshow('Original', gs_img)
cv2.imshow('Noise', noise_img)

cv2.waitKey(0)

cv2.imshow('ADF', adf_img)
cv2.imshow('Gauss', gauss_img)

# 4
print('ADF:')
print(f'MSE: {qm.mse(gs_img, adf_img)}')
print(f'PSNR: {qm.PSNR(gs_img, adf_img)}')
print('\nGauss:')
print(f'MSE: {qm.mse(gs_img, gauss_img)}')
print(f'PSNR: {qm.PSNR(gs_img, gauss_img)}')

cv2.waitKey(0)
cv2.destroyWindow('Original')
cv2.destroyWindow('Noise')

cv2.imshow('Sobel in ADF', apply_sobel(adf_img))
cv2.imshow('Sobel in Gauss', apply_sobel(gauss_img))

cv2.waitKey(0)
cv2.destroyWindow('Sobel in ADF')
cv2.destroyWindow('Sobel in Gauss')

um_adf_img = apply_UMHF(adf_img, 1)
cv2.imshow('ADF with UM', um_adf_img)
hf_adf_img = apply_UMHF(adf_img, 4)
cv2.imshow('ADF with HF', hf_adf_img)

um_gauss_img = apply_UMHF(gauss_img, 1)
hf_gauss_img = apply_UMHF(gauss_img, 4)
cv2.imshow('Gauss with UM', um_gauss_img)
cv2.imshow('Gauss with HF', hf_gauss_img)

cv2.waitKey(0)
cv2.destroyWindow('ADF with UM')
cv2.destroyWindow('ADF with HF')
cv2.destroyWindow('Gauss with UM')
cv2.destroyWindow('Gauss with HF')

cv2.imshow('Sobel in ADF with UM', apply_sobel(um_adf_img))
cv2.imshow('Sobel in Gauss with UM', apply_sobel(um_gauss_img))

cv2.imshow('Sobel in ADF with HF', apply_sobel(hf_adf_img))
cv2.imshow('Sobel in Gauss with HF', apply_sobel(hf_gauss_img))

cv2.waitKey(0)
cv2.destroyAllWindows()
