from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

image_tensor=torch.load("./rgb_info.pth").cpu()
image_tensor_scaled = image_tensor.float() * 255
flipped_image_tensor = torch.flip(image_tensor_scaled, [0])

# Convert the flipped tensor to a numpy array for plotting, ensuring it's in uint8 format.
flipped_image_array = flipped_image_tensor.numpy().astype(np.uint8)

# Create a new figure and axis for plotting the image.
fig, ax = plt.subplots()

# Plot the flipped image.
ax.imshow(flipped_image_array)

# Set the labels for the axes as per the user's description.
ax.set_xlabel('Y-axis')
ax.set_ylabel('X-axis')

# Set the ticks such that x-axis numbers increase from top to bottom and y-axis numbers increase from left to right.
# Since the image is flipped, the top of the image is now the bottom, so we reverse the y-ticks to match this.
ax.set_xticks(np.arange(0, flipped_image_array.shape[1], 50))
ax.set_yticks(np.arange(flipped_image_array.shape[0], 0, -50))

# Ensure the origin is at the top left as per the user's instructions.
# Invert the y-axis to make the x-axis go from 0 at the top to 250 at the bottom after flipping the image.
ax.invert_yaxis()

# Set the aspect of the plot to equal, so the image isn't stretched.
ax.set_aspect('equal')

# Remove the padding around the image.
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
output_path = './rendered_image.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)