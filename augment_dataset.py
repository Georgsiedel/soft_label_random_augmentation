import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from typing import Optional, List
import os

import torchvision
from torchvision import datasets, transforms

from augmentations.trivial_augment import CustomTrivialAugmentWide
from augmentations.random_crop import RandomCrop
from augmentations.random_erasing import RandomErasing
from utils.display_image import display_image_grid
import random
from torch.utils.data import Dataset, ConcatDataset
    
class CustomDataset(Dataset):
    def __init__(self, np_images, original_dataset):
        # Load images
        self.images = torch.from_numpy(np_images).permute(0, 3, 1, 2) / 255
        #Normalize the images
        #transform_test = transforms.Compose([
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #])
        #self.images = transform_test(self.images)
        
        # Extract labels from the original PyTorch dataset
        self.labels = [label for _, label in original_dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.images[index]
        label = self.labels[index]

        return image, label

class Placeholder_with_Confidence:
    def __init__(self, default_confidence=1.0, default_magnitude = -1):
        self.default_magnitude = default_magnitude
        self.default_conf = default_confidence

    def __call__(self, img):
        return img, self.default_magnitude, self.default_conf


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

        prep_x = self.preprocess(x)

        augment_x, augmentation_magnitude, confidences = self.transforms_augmentation(prep_x)
        if isinstance(confidences, tuple):
            combined_confidence = self.get_confidence(confidences)
        elif isinstance(confidences, float):
            combined_confidence = confidences
        else:
            raise TypeError(f"Expected float or tuple of confidences but got {type(confidences)}")

        return augment_x, y, combined_confidence, augmentation_magnitude

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
        if dataset_name == "TinyImageNet":
            augmentations.append(transforms.RandomCrop(64, padding=4))
        else:
            augmentations.append(transforms.RandomCrop(32, padding=4))

    if trivial_augment == 0:
        augmentations.append(Placeholder_with_Confidence(1.0, -1))
    elif trivial_augment == 1:
        augmentations.append(CustomTrivialAugmentWide(
                    soft=False,
                    severity=augmentation_severity,
                    selected_transforms=selected_transforms,
                    get_signed=augmentation_sign,
                    dataset_name=dataset_name,
                    mapping_approach=mapping_approach,
                ))
    elif trivial_augment == 2:
        augmentations.append(CustomTrivialAugmentWide(
                    soft=True,
                    severity=augmentation_severity,
                    selected_transforms=selected_transforms,
                    get_signed=augmentation_sign,
                    dataset_name=dataset_name,
                    mapping_approach=mapping_approach,
                ))
    
    if random_cropping == 2:
        augmentations.append(RandomCrop(dataset_name=dataset_name, custom=True))
    elif random_cropping == 3:
        augmentations.append(RandomCrop(dataset_name=dataset_name, custom=False))

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
        base_trainset = datasets.CIFAR10(root="../data", train=True, download=True)
        testset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transforms_preprocess)
        num_classes = 10
        factor = 1
        train_workers = 1
    elif dataset_name == "CIFAR100":
        # CIFAR-100
        base_trainset = datasets.CIFAR100(root="../data", train=True, download=True)
        testset = datasets.CIFAR100(root="../data", train=False, download=True, transform=transforms_preprocess)
        num_classes = 100
        factor = 1
        train_workers = 1
    elif dataset_name == "TinyImageNet":
        base_trainset = datasets.ImageFolder(root="../data/TinyImageNet/train")
        testset = datasets.ImageFolder(root="../data/TinyImageNet//val", transform=transforms_preprocess)
        num_classes = 200
        factor = 2
        train_workers = 2
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    trainset = AugmentedDataset(
        dataset=base_trainset,
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
    )

    return trainset, testset, num_classes, factor, train_workers


# Define the function to load corrupted datasets separately
def load_data_c_separately(dataset, testset, batch_size, transforms_preprocess):
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    c_datasets = {}
    for corruption in corruptions:
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            np_data_c = np.load(f'../data/{dataset}-c/{corruption}.npy')
            np_data_c = np.array(np.array_split(np_data_c, 5))
            custom_dataset = CustomDataset(np_data_c[0], testset)  # Load only one split for now
            custom_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
            c_datasets[corruption] = custom_dataloader
        elif dataset == 'TinyImageNet':
            intensity_datasets = [datasets.ImageFolder(root=os.path.abspath(f'../data/TinyImageNet-c/' + corruption + '/' + str(intensity)),
                                                                        transform=transforms_preprocess) for intensity in range(1, 6)]
            concat_intensities = ConcatDataset(intensity_datasets)
            custom_dataloader = torch.utils.data.DataLoader(concat_intensities, batch_size=batch_size, shuffle=False)
            c_datasets[corruption] = custom_dataloader
        else:
            print('No corrupted benchmark available other than CIFAR10-c.')

    return c_datasets, corruptions

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize(seed=0, batch_size=256, selected_transforms=None, augmentation_sign=False,
              augmentation_severity=-1, dataset="CIFAR10", random_cropping=1, trivial_augment=0,
              random_erasing=0, random_erasing_p=0.3, random_erasing_max_scale=0.33):
    
    g = torch.Generator()
    g.manual_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transforms_preprocess, transforms_augmentation = create_transforms(
        trivial_augment=trivial_augment, random_erasing=random_erasing, random_erasing_p=random_erasing_p,
        random_erasing_max_scale=random_erasing_max_scale, random_cropping=random_cropping,
        selected_transforms=selected_transforms, augmentation_severity=augmentation_severity,
        augmentation_sign=augmentation_sign, dataset_name=dataset, seed=seed
    )

    trainset, testset, num_classes = load_data(transforms_preprocess, transforms_augmentation, dataset_name=dataset)
    print(f"Transforms: {transforms_augmentation}")
    print(f"Number of classes: {num_classes}")
    
    # Create a data loader for the training set
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              worker_init_fn=seed_worker, 
                                              generator=g)

    # Display a grid of images with labels and confidence scores
    images, labels, confidences = next(iter(trainloader))
    display_image_grid(images, labels, confidences, batch_size=64, classes=num_classes)
    #print(f"Confidence: {confidences}")

if __name__ == "__main__":
    visualize()
