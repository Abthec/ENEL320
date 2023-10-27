import cv2
import scipy as sp
import numpy as np
import matplotlib as plt

image_path = "blur_car.png"
image = cv2.imread(image_path, 0)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shft = np.fft.fftshift(dft)
magnitude = 20*np.log(cv2.magnitude(dft_shft[:,:,0], dft_shft[:,:,1]))

magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

print(image)
print(dft)


cv2.imshow('Discrete Fourier Transform', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
