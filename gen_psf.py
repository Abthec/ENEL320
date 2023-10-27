import cv2
import numpy as np

def generate_line_psf(kernel_size, line_thickness, border):
    psf = np.zeros((kernel_size, kernel_size, 3), dtype=np.float32)
    center = kernel_size // 2
    start = center - line_thickness // 2
    end = center + line_thickness // 2

    # Create a white line within the specified border
    psf[start:end, border:-border] = 1.0

    return psf

# Define the size of the PSF, the line thickness, and the border width
kernel_size = 480  # Adjust the kernel size as needed
line_thickness = 4  # Adjust the line thickness as needed
border_width = 200  # Adjust the border width as needed

# Generate the line PSF
psf = generate_line_psf(kernel_size, line_thickness, border_width)

# Normalize the PSF to 0-255 range for display
psf_normalized = cv2.normalize(psf, None, 0, 255, cv2.NORM_MINMAX)

print(psf_normalized)

# Display the PSF as an image
cv2.imshow('Line PSF', psf_normalized.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()