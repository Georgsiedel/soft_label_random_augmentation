import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from typing import Optional, List

import torchvision
from torchvision import datasets, transforms

from augmentations.trivial_augment import CustomTrivialAugmentWide
from augmentations.random_crop import RandomCrop
from augmentations.random_erasing import RandomErasing
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Error: Boolean value expected for argument {v}.')

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions.

    Attributes:
        dataset (torch.utils.data.Dataset): The base dataset to augment.
        transforms_preprocess (transforms.Compose): Transformations for preprocessing.
        transforms_augmentation (transforms.Compose): Transformations for augmentation.
    """

    def __init__(
        self,
        dataset,
        transforms_preprocess,
        transforms_augmentation,
    ):
        if dataset is not None:
            self.dataset = dataset

        self.preprocess = transforms_preprocess
        self.transforms_augmentation = transforms_augmentation
        self.original_length = getattr(dataset, "original_length", None)
        self.generated_length = getattr(dataset, "generated_length", None)

    def get_confidence(self, confidences: Optional[tuple]) -> Optional[float]:
        """Combines multiple confidence values into a single value.

        Args:
            confidences (Optional[tuple]): A tuple of confidence values.

        Returns:
            Optional[float]: The combined confidence value.
        """
        combined_confidence = reduce(lambda x, y: x * y, confidences)
        # print(f"Confidences: {confidences}\tCombined Confidence: {combined_confidence}\n")
        return combined_confidence

    def __getitem__(self, i: Optional[int]) -> Optional[tuple]:
        """Retrieves an item from the dataset and applies augmentations.

        Args:
            i (Optional[int]): Index of the item to retrieve.

        Returns:
            Optional[tuple]: The augmented image, the label, and the combined confidence value.
        """
        x, y = self.dataset[i]
        confidences = None
        augmentation_magnitude = None
        combined_confidence = torch.tensor(1.0, dtype=torch.float32)

        augment = self.transforms_augmentation

        if isinstance(x, tuple):
            raise ValueError("Tuple not supported")
        
        augment_x = augment(x)

        if isinstance(augment_x, tuple):
            confidences = augment_x[1]
            augment_x = augment_x[0]
            if isinstance(confidences, tuple):
                combined_confidence = self.get_confidence(confidences)
            elif isinstance(confidences, list):
                combined_confidence = confidences[1]
                augmentation_magnitude = confidences[0]
            else:
                combined_confidence = confidences

        if not isinstance(augment_x, torch.Tensor):
            augment_x = self.preprocess(augment_x)

        if augmentation_magnitude is not None:
            return augment_x, y, [augmentation_magnitude, combined_confidence]
        return augment_x, y, combined_confidence

    def __len__(self):
        return len(self.dataset)


def create_transforms(
    trivial_augment: int = 0,
    random_erasing: int = 0,
    random_erasing_p: float = 0.3,
    random_erasing_max_scale: float = 0.33,
    random_cropping: int = 0,
    selected_transforms: Optional[List[str]] = None,
    augmentation_severity: int = 0,
    augmentation_sign: bool = False,
    dataset_name: str = "CIFAR10",
    seed: Optional[int] = None,
    individual_analysis: Optional[bool] = False,
    mapping_approach: Optional[str] = "exact_model_accuracy",
) -> Optional[tuple]:
    """Creates preprocessing and augmentation transformations.

    Args:
    trivial_augment (int): 0: no TA; 1: standard TA; 2: soft TA.
        random_erasing (int): 0: no RE; 1: standard RE; 2: soft RE.
        random_cropping  (int): 0: no RC; 1: standard RC; 2: soft RC.
        augmentation_name (str, optional): Name of the custom augmentation (if applicable).
        augmentation_severity (int, optional): Severity level for custom augmentations. Defaults to 0.
        augmentation_sign (bool, optional): Flag to determine if augmentation should be signed. Defaults to False.
        dataset_name (str, optional): Name of the dataset. Defaults to "CIFAR10".
        seed (int, optional): Random seed for reproducibility.
        individual_analysis (bool, optional): Whether to perform individual analysis of augmentations.
        mapping_approach (str, optional): Approach for mapping confidence. Defaults to "exact_model_accuracy".

    Returns:
        Optional[tuple]: The preprocessing and augmentation transformations.
    """
    augmentations = [
        transforms.RandomHorizontalFlip()       # For Tiny-ImageNet: 64 x 64; For CIFAR: 32 x 32
    ]

    if random_cropping == 1:
        augmentations.append(transforms.RandomCrop(32, padding=4))
    elif random_cropping == 2:
        augmentations.append(RandomCrop(dataset_name=dataset_name, custom=True, seed=seed))

    if trivial_augment == 1:
        augmentations.append(CustomTrivialAugmentWide(
                    soft=False,
                    severity=augmentation_severity,
                    selected_transforms=selected_transforms,
                    get_signed=augmentation_sign,
                    dataset_name=dataset_name,
                    individual_analysis=individual_analysis,
                    mapping_approach=mapping_approach,
                ))
    elif trivial_augment == 2:
        augmentations.append(CustomTrivialAugmentWide(
                    soft=True,
                    severity=augmentation_severity,
                    selected_transforms=selected_transforms,
                    get_signed=augmentation_sign,
                    dataset_name=dataset_name,
                    individual_analysis=individual_analysis,
                    mapping_approach=mapping_approach,
                ))

        print(f"Calculating confidence with the mapping approach: {mapping_approach}\n")

    if random_erasing == 1:
        augmentations.append(RandomErasing(p=random_erasing_p, scale=(0.02, random_erasing_max_scale), ratio=(0.3, 3.3), value='random', custom=False, dataset_name=dataset_name))
    elif random_erasing == 2:
        augmentations.append(RandomErasing(p=random_erasing_p, scale=(0.02, random_erasing_max_scale), ratio=(0.3, 3.3), value='random', custom=True, dataset_name=dataset_name))

    transforms_preprocess = transforms.ToTensor()
    transforms_augmentation = transforms.Compose(augmentations)

    return transforms_preprocess, transforms_augmentation


def load_data(
    transforms_preprocess,
    transforms_augmentation=None,
    dataset_name: Optional[str] = "CIFAR10",
) -> Optional[tuple]:
    """Loads and prepares a dataset (CIFAR-10, CIFAR-100, or Tiny-ImageNet) with specified transformations.

    Args:
        transforms_preprocess (transforms.Compose): Preprocessing transformations.
        transforms_augmentation (transforms.Compose, optional): Augmentation transformations.
        dataset_split (int or str, optional): Number of samples to retain for faster testing. 
            If "full", the entire dataset is used.
        dataset_name (str, optional): Name of the dataset to load. Supports "CIFAR10", "CIFAR100", and "Tiny-ImageNet".

    Returns:
        Optional[tuple]: The processed training and testing datasets.
    """

    if dataset_name == "CIFAR10":
        # CIFAR-10
        base_trainset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transforms_preprocess)
        base_testset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transforms_preprocess)
    elif dataset_name == "CIFAR100":
        # CIFAR-100
        base_trainset = datasets.CIFAR100(root="../data", train=True, download=True, transform=transforms_preprocess)
        base_testset = datasets.CIFAR100(root="../data", train=False, download=True, transform=transforms_preprocess)
    elif dataset_name == "TinyImageNet":
        base_trainset = datasets.ImageFolder(root="../data/TinyImageNet/train", transform=transforms_preprocess)
        base_testset = datasets.ImageFolder(root="../data/TinyImageNet//val", transform=transforms_preprocess)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


    trainset = AugmentedDataset(
        dataset=base_trainset,
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
    )

    testset = AugmentedDataset(
        dataset=base_testset,
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
    )

    return trainset, testset


def display_image_grid(images, labels, confidences, batch_size, classes):
    """
    Displays a 5x5 grid of images with labels and confidence scores.

    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Corresponding labels for the images.
        confidences (torch.Tensor): Corresponding confidence scores for the images.
        batch_size (int): Number of images to display in the grid (should be 25 for a 5x5 grid).
        classes (list): List of class names for labeling.
    """
    # Limit batch_size to 25 for a 5x5 grid
    batch_size = min(batch_size, 25)
    
    if isinstance(confidences, list):
        confidences = confidences[1]

    # Convert images to a grid, with 5 images per row
    grid_img = torchvision.utils.make_grid(images[:batch_size], nrow=5)

    # Convert from tensor to numpy for display
    npimg = grid_img.numpy()

    # Plot the grid with appropriate figure size (for 5x5 grid)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

    # Add titles for each image (labels and confidence scores)
    for i in range(batch_size):
        ax = plt.subplot(5, 5, i + 1)  # Adjust to a 5x5 grid
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        ax.set_title(
            f"{labels[i].item()} ({classes[labels[i].item()]})\nConf: {confidences[i]:.2f}",
            fontsize=8
        )
        ax.axis("off")

    plt.show()

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.seed(worker_seed)



import argparse
parser = argparse.ArgumentParser(description='Soft Augmentations')
parser.add_argument('--individual_analysis', type=str2bool, nargs='?', const=False, default=False,
                    help='individual transforms of TrivialAugment')
parser.add_argument('--seed', default=0, type=int, help='seed number')
parser.add_argument('--selected_transforms', type=str, nargs='+', default=None,
                    help="List of TA transforms for individual analysis, will be applied if given")
parser.add_argument('--augmentation_sign', type=str2bool, nargs='?', const=False, default=False,
                    help='individual transforms of TrivialAugment')
parser.add_argument('--augmentation_severity', default=-1, type=int, help='severity for an individual TA analysis')

args = parser.parse_args()


if __name__ == "__main__":

    # Set the batch size for the data loader
    batch_size = 256
    DATASET_NAME = "CIFAR10"

    # Set the random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Mapping approach for confidence calculation
    """
    Mapping Approaches:
    1. exact_model_accuracy
    2. smoothened_hvs
    3. fixed_params
    4. exact_hvs
    5. ssim_metric
    6. uiq_metric
    7. ncc_metric
    8. scc_metric
    9. sift_metric
    """
    # Create the transformations for preprocessing and augmentation
    transforms_preprocess, transforms_augmentation = create_transforms(random_cropping=1,
                                                                       trivial_augment=0,
                                                                       random_erasing=0,
                                                                       random_erasing_p=0.3,
                                                                       random_erasing_max_scale=0.33,
                                                                       selected_transforms=args.selected_transforms,
                                                                       augmentation_severity=args.augmentation_severity, 
                                                                       augmentation_sign=args.augmentation_sign, 
                                                                       dataset_name=DATASET_NAME,
                                                                       seed=args.seed,
                                                                       individual_analysis=args.individual_analysis,
                                                                       mapping_approach="fixed_params")
    
    print(transforms_augmentation)

    # Load the CIFAR-10 dataset with the specified transformations
    trainset, testset = load_data(transforms_preprocess=transforms_preprocess, 
                                  transforms_augmentation=transforms_augmentation, 
                                  dataset_name=DATASET_NAME)

    # Create a data loader for the training set
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              worker_init_fn=seed_worker, 
                                              generator=g)

    # Display a grid of images with labels and confidence scores
    classes = trainset.dataset.classes
    images, labels, confidences = next(iter(trainloader))
    # display_image_grid(images, labels, confidences, batch_size=batch_size, classes=classes)
    print(f"Confidence: {confidences}")

