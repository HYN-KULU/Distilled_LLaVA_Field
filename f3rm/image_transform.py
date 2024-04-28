from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

image_tensor=torch.load("./rgb_info.pth").cpu()
image_tensor_scaled = image_tensor.float() * 255
image_array_rgb = image_tensor_scaled .numpy().astype(np.uint8)
image_array_symmetric = np.flip(image_array_rgb, axis=0)
fig, ax = plt.subplots()
ax.imshow(image_array_symmetric)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.invert_yaxis()
ax.set_xticks(np.arange(0, image_array_symmetric.shape[1], 50))
ax.set_yticks(np.arange(0, image_array_symmetric.shape[0], 50))
ax.set_aspect('equal')
plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
plt.tight_layout()

fig.canvas.draw()

# Convert the figure canvas to an array
image_plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_plot_array = image_plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# Instead of trying to set the writable flag, create a copy of the array directly when converting to a tensor
image_plot_tensor = torch.tensor(image_plot_array.copy()) 
image_np = image_plot_tensor.numpy()

# Create a PIL image from the numpy array
image_pil = Image.fromarray(image_np)

image_pil.save("./converted_image.png")
# output_path = './rendered_image.png'
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0)