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
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import ToPILImage


def normalized_cross_correlation(im1: torch.Tensor, im2: torch.Tensor) -> float:
    """
    Compute normalized cross-correlation between two image tensors.
    Args:
        im1, im2: torch.Tensor of shape (C, H, W) or (H, W), float dtype.
    Returns:
        Normalized NCC value in [0,1].
    """
    # Ensure shape (1, H, W)
    if im1.dim() == 3:
        im1_gray = rgb_to_grayscale(im1.unsqueeze(0)).squeeze(0)
        im2_gray = rgb_to_grayscale(im2.unsqueeze(0)).squeeze(0)
    else:
        im1_gray = im1
        im2_gray = im2

    # Convert to numpy arrays
    arr1 = im1_gray.cpu().numpy().astype(np.float32)
    arr2 = im2_gray.cpu().numpy().astype(np.float32)

    # Normalize
    arr1 = (arr1 - arr1.mean()) / (arr1.std() + 1e-4)
    arr2 = (arr2 - arr2.mean()) / (arr2.std() + 1e-4)

    # Correlate
    ncc_mat = correlate2d(arr1, arr2, mode="valid")
    ncc_val = np.max(ncc_mat) / arr1.size

    return float((ncc_val + 1.0) / 2.0)


def multiscale_structural_similarity(
    im1: torch.Tensor,
    im2: torch.Tensor
) -> float:
    """
    Compute MS-SSIM structural similarity component.
    Args:
        im1, im2: torch.Tensor of shape (C, H, W), values in [0,1].
    Returns:
        Structural similarity index measure (float).
    """
    # Prepare batch dimension
    x1 = im1.unsqueeze(0).float()
    x2 = im2.unsqueeze(0).float()

    # compute the structural similarity index
    ssim = StructuralSimilarityIndexMeasure(return_contrast_sensitivity=True)
    structural_value, contrast_value = ssim(im1, im2)
    luminance_value = structural_value / contrast_value
    return structural_value.item()  # structural_value
    # return luminance_value.item()     # luminance_value


def multiscale_contrast_similarity(
    im1: torch.Tensor,
    im2: torch.Tensor
) -> float:
    """
    Compute MS-SSIM contrast sensitivity component.
    """
    x1 = im1.unsqueeze(0).float()
    x2 = im2.unsqueeze(0).float()

    ssim = StructuralSimilarityIndexMeasure(return_contrast_sensitivity=True)
    struct_val, contrast_val = ssim(x1, x2)
    return float(contrast_val)


def spatial_correlation_coefficient(
    im1: torch.Tensor,
    im2: torch.Tensor
) -> float:
    """
    Compute spatial correlation coefficient (SCC) between two images.
    """
    x1 = im1.unsqueeze(0).float()
    x2 = im2.unsqueeze(0).float()

    scc = SpatialCorrelationCoefficient()
    scc_val = scc(x1, x2)
    # Map from [-1,1] to [0,1]
    return float((scc_val + 1.0) / 2.0)


def universal_image_quality_index(
    im1: torch.Tensor,
    im2: torch.Tensor
) -> float:
    """
    Compute Universal Image Quality Index (UIQ).
    """
    x1 = im1.unsqueeze(0).float()
    x2 = im2.unsqueeze(0).float()

    uiq = UniversalImageQualityIndex()
    uiq_val = uiq(x1, x2)
    return float(uiq_val)


def sift_correction_factor(
    original: torch.Tensor,
    augmented: torch.Tensor,
    display_matches: bool = False
) -> float:
    """
    Compute SIFT-based correction factor between original and augmented images.
    Args:
        original, augmented: torch.Tensor of shape (C, H, W), values [0,1]
    """
    # Convert to uint8 grayscale numpy for SIFT
    to_pil = ToPILImage()
    pil_orig = to_pil(original)
    pil_aug = to_pil(augmented)

    # Perform SIFT operations (expect PIL or numpy inside)
    matches_ref = sift_operation(pil_orig, pil_orig)
    matches_oa = sift_operation(pil_orig, pil_aug, display_matches)

    return matches_oa / matches_ref
