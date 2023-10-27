import cv2
import numpy as np


def resize_image(image, psf):
    """
  Resize an image to match the size of a point spread function (PSF).

  Args:
    image: A numpy array representing the input image.
    psf: A numpy array representing the point spread function.

  Returns:
    A numpy array representing the resized image.
  """

    # Get the size of the image
    psf_height, psf_width = image.shape[:2]

    # Resize the image to match the PSF size.
    resized_image = cv2.resize(psf, (psf_width, psf_height), interpolation=cv2.INTER_LINEAR)

    return resized_image


def wiener_deblurring(image, psf, K=0.02):

    # Convert the image and PSF to the Fourier domain.
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf)

    # Calculate the Wiener filter.
    W = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + (K * np.ones_like(psf_fft)))

    # Apply the Wiener filter to the image.
    deblurred_image_fft = image_fft * W

    # Convert the deblurred image back to the spatial domain.
    deblurred_image = np.fft.ifft2(deblurred_image_fft)

    return deblurred_image


def generate_line_psf(kernel_size, line_thickness, border):
    psf = np.zeros((kernel_size, kernel_size, 3), dtype=np.float32)
    center = kernel_size // 2
    start = center - line_thickness // 2
    end = center + line_thickness // 2

    psf[start:end, border:-border] = 1.0

    psf = cv2.normalize(psf, None, 0, 255, cv2.NORM_MINMAX)

    return psf



image = cv2.imread("blur_car.png")
# psf = cv2.imread("psf1.png")

kernel_size = 480
line_thickness = 4
border_width = 200

# Generate the line PSF
psf = generate_line_psf(kernel_size, line_thickness, border_width)

resized_psf = resize_image(image, psf)

deblur_image = wiener_deblurring(image, resized_psf)

cv2.imshow('Blurred Image', image)
cv2.imshow('PSFFFT', np.fft.fft2(resized_psf).astype(np.uint8))
cv2.imshow('PSF', resized_psf)
cv2.imshow('Deblurred Image', deblur_image.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
