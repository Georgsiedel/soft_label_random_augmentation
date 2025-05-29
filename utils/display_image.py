import matplotlib.pyplot as plt
import torch

def display_image_grid(images, labels, confidences, augmentation_types, batch_size, classes):
    """
    Displays a grid of images with labels and confidence scores.

    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Corresponding labels for the images.
        confidences (torch.Tensor): Corresponding confidence scores for the images.
        batch_size (int): Number of images to display in the grid (should be 25 for a 5x5 grid).
        classes (list): List of class names for labeling.
    """
    # Limit batch_size for the grid
    batch_size = min(batch_size, 15)
    
    # Ensure confidences are correctly formatted as a tensor
    if isinstance(confidences, list):
        confidences = torch.tensor(confidences)

    plt.figure(figsize=(6.4,4.5))
    for i in range(batch_size):
        ax = plt.subplot(3, 5, i + 1)
        # Convert image to numpy for display
        img = images[i].permute(1, 2, 0).numpy()  # From (C, H, W) to (H, W, C)
        ax.imshow(img)
        ax.set_title(
            f"{augmentation_types[i]}\n{classes[labels[i].item()]} ({confidences[i]:.2f})",
            fontsize=8, pad=3
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_images(original_image, augmented_image):
    # Convert images to displayable format
    original_image_np = (
        original_image[0].permute(1, 2, 0).numpy()
    )  # Convert CHW to HWC format
    augmented_image_np = (
        augmented_image[0].permute(1, 2, 0).numpy()
    )  # Convert CHW to HWC format

    # Plot the original and augmented images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original_image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(augmented_image_np)
    axs[1].set_title("Augmented Image")
    axs[1].axis("off")

    plt.show()
