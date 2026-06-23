import matplotlib.pyplot as plt
import numpy as np
import os


def save_tensor_images(tensor, save_dir, num_images=5, prefix='image', figsize=(15, 15)):
    """
    Save images from a tensor with shape (batch_size, 3, height, width) to the specified directory.

    Parameters:
    - tensor: PyTorch tensor or NumPy array with shape (batch_size, 3, height, width)
    - save_dir: Directory where the images will be saved
    - num_images: Number of images to save
    - prefix: Prefix for the saved image filenames
    - figsize: Size of the figure
    """
    # Move the tensor to CPU if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to NumPy array if it's a PyTorch tensor
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.numpy()

    # Ensure the tensor has the correct shape
    assert tensor.ndim == 4 and tensor.shape[
        1] == 3, "Tensor shape must be (batch_size, 3, height, width)"

    # Determine the number of images to save
    num_images = min(num_images, tensor.shape[0])

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):
        # Create a new figure
        fig, ax = plt.subplots(figsize=figsize)

        # Select the image and transpose to (height, width, 3)
        img = tensor[i].transpose(1, 2, 0)

        # Debug: Print image statistics before normalization
        # print(f"Image {i} - min: {img.min()}, max: {img.max()}")

        # Normalize the image to the range [0, 1]
        img = (img - img.min()) / (img.max() - img.min())

        # Clip the image to the range [0, 1] to ensure valid range
        img = np.clip(img, 0, 1)

        # Debug: Print image statistics after normalization
        # print(f"Normalized Image {i} - min: {img.min()}, max: {img.max()}")

        # Display the image
        ax.imshow(img)
        ax.axis('off')

        # Save the figure
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory


# Example usage:
# Assuming you have a tensor 'images' with shape (batch_size, 3, height, width)
# save_tensor_images(images, save_dir='output_images', num_images=5)
