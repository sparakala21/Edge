import numpy as np
from PIL import Image

# Read the image and convert to grayscale
image = Image.open('images/sky.jpg').convert('L')
image_array = np.array(image)

gaussian_smoothing = np.array([[2, 4, 5, 4, 2],
                               [4, 9, 12, 9, 4],
                               [5, 12, 15, 12, 5],
                               [4, 9, 12, 9, 4],
                               [2, 4, 5, 4, 2]]) / 159

# Get the dimensions of the image
image_height, image_width = image_array.shape

# Create output arrays
gaussian_result = np.zeros_like(image_array)

# Apply Gaussian smoothing
for i in range(2, image_height-2):
    for j in range(2, image_width-2):
        region = image_array[i-2:i+3, j-2:j+3]
        gaussian_result[i, j] = np.sum(region * gaussian_smoothing)



gaussian_image = Image.fromarray(gaussian_result)

# Save or display the output images
gaussian_image.save('images/gaussian_sky.jpg')