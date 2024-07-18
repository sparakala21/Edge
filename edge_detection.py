import numpy as np
from PIL import Image

# Read the image and convert to grayscale
image = Image.open('images/church_interior.jpg').convert('L')
image_array = np.array(image)

gaussian_smoothing = np.array([[2, 4, 5, 4, 2],
                               [4, 9, 12, 9, 4],
                               [5, 12, 15, 12, 5],
                               [4, 9, 12, 9, 4],
                               [2, 4, 5, 4, 2]]) / 159

sobel_operator_h = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
sobel_operator_v = sobel_operator_h.T

# Get the dimensions of the image
image_height, image_width = image_array.shape

# Create output arrays
gaussian_result = np.zeros_like(image_array)
sobel_h_result = np.zeros_like(image_array)
sobel_v_result = np.zeros_like(image_array)

# Apply Gaussian smoothing
for i in range(2, image_height-2):
    for j in range(2, image_width-2):
        region = image_array[i-2:i+3, j-2:j+3]
        gaussian_result[i, j] = np.sum(region * gaussian_smoothing)

# Apply Sobel operator (horizontal)
for i in range(1, image_height-1):
    for j in range(1, image_width-1):
        region = gaussian_result[i-1:i+2, j-1:j+2]
        sobel_h_result[i, j] = np.sum(region * sobel_operator_h)

# Apply Sobel operator (vertical)
for i in range(1, image_height-1):
    for j in range(1, image_width-1):
        region = gaussian_result[i-1:i+2, j-1:j+2]
        sobel_v_result[i, j] = np.sum(region * sobel_operator_v)

# Calculate gradient magnitude and phase
gradient = np.sqrt(sobel_h_result**2 + sobel_v_result**2)
phase = np.arctan2(sobel_h_result, sobel_v_result) * (180.0 / np.pi)
phase = ((45 * np.round(phase / 45.0)) + 180) % 180
phase = phase.astype(np.uint8)

# Apply Gaussian smoothing to the phase result
phase_smoothed = np.zeros_like(phase)
for i in range(2, image_height-2):
    for j in range(2, image_width-2):
        region = phase[i-2:i+3, j-2:j+3]
        phase_smoothed[i, j] = np.sum(region * gaussian_smoothing)

# Normalize gradient to the range [0, 255]
gradient = (gradient / gradient.max()) * 255
gradient = gradient.astype(np.uint8)

# Convert the output array back to images
phase_smoothed_image = Image.fromarray(phase_smoothed)
gradient_image = Image.fromarray(gradient)

# Save or display the output images
phase_smoothed_image.save('results/phase_smoothed_church_interior.jpg')
gradient_image.save('results/gradient_church_interior.jpg')

phase_smoothed_image.show()
gradient_image.show()
