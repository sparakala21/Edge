import numpy as np
from PIL import Image

# Read the image and convert to grayscale
image = Image.open('images/mattdamon.jpg').convert('L')
image_array = np.array(image)

ridge_detection = np.array([[0, -1, 0],
                           [-1, 4, 1],
                           [0, -1, 0]])

edge_detection = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

prewitt_detection = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
# Define the 3x3 kernel (example: a simple blurring kernel)
box_blur = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0

# Get the dimensions of the image
image_height, image_width = image_array.shape

# Create an output array
output_array = np.zeros_like(image_array)

# Apply the kernel to each pixel in the image
for i in range(1, image_height-1):
    for j in range(1, image_width-1):
        # Extract the 3x3 region
        region = image_array[i-1:i+2, j-1:j+2]
        # Apply the kernel (element-wise multiplication and sum)
        output_array[i, j] = np.sum(region * edge_detection)

# Convert the output array back to an image
output_image = Image.fromarray(output_array)

# Save or display the output image
output_image.save('images/prewittdamon.jpg')
output_image.show()
