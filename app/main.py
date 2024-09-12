import cv2
import numpy as np
import anisotropic_diffusion_filter as adf
import noise as ns

gs_img = cv2.imread('images/image.png', 0)

noise_img = ns.add_gauss_noise(gs_img)
# noise_img = ns.add_salt_pepper_noise(img_grayscale_basic, 0.05, 0.05)
# noise_img = ns.add_poisson_noise(img_grayscale_basic)
# noise_img = ns.add_speckle_noise(img_grayscale_basic)

adf_image = adf.adf_convolution(noise_img, 0.25, 0.25, 10)

cv2.imshow('Grayscale Image', gs_img)
cv2.imshow('Noise Image', noise_img)
cv2.imshow('Adf Image', adf_image)


cv2.waitKey(0)
cv2.destroyAllWindows()