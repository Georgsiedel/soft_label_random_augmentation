import numpy as np
from PIL import Image
from scipy.signal import correlate2d
from torchvision import transforms

from utils.sift_comparison import sift_operation
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    SpatialCorrelationCoefficient,
    UniversalImageQualityIndex,
)


def normalized_cross_correlation(im1: Image.Image, im2: Image.Image):
    # convert images to grayscale numpy arrays
    im1_np = np.array(im1.convert("L"), dtype=np.float32)
    im2_np = np.array(im2.convert("L"), dtype=np.float32)

    # normalize the images
    im1_np = (im1_np - np.mean(im1_np)) / (np.std(im1_np) + 1e-4)
    im2_np = (im2_np - np.mean(im2_np)) / (np.std(im2_np) + 1e-4)

    # compute the normalized cross correlation
    ncc_matrix = correlate2d(im1_np, im2_np, mode="valid")
    ncc_value = np.max(ncc_matrix) / (im1_np.size)

    return (ncc_value + 1.0) / 2.0


def multiscale_structural_similarity(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the structural similarity index
    ssim = StructuralSimilarityIndexMeasure(return_contrast_sensitivity=True)
    structural_value, contrast_value = ssim(im1, im2)
    luminance_value = structural_value / contrast_value
    return structural_value.item()  # structural_value
    # return luminance_value.item()     # luminance_value


def multiscale_contrast_similarity(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the structural similarity index
    ssim = StructuralSimilarityIndexMeasure(return_contrast_sensitivity=True)
    structural_value, contrast_value = ssim(im1, im2)
    return contrast_value.item()  # contrast_value


def spatial_correlation_coefficient(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the spatial correlation coefficient
    scc = SpatialCorrelationCoefficient()
    spatial_correlation_value = scc(im1, im2)
    return (spatial_correlation_value.item() + 1.0) / 2.0


def universal_image_quality_index(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the universal image quality index
    uiq = UniversalImageQualityIndex()
    uiq_value = uiq(im1, im2)
    return uiq_value.item()

def sift_correction_factor(original_image, augmented_image, display_matches: bool = False):
    matches_reference = sift_operation(original_image, original_image)
    matches_12 = sift_operation(original_image, augmented_image, display_matches)
    correction_factor = matches_12 / matches_reference
    return correction_factor