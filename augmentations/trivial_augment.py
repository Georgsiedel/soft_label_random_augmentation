import torch
from PIL import Image
from typing import Optional
import numpy as np
import os
import pandas as pd

from utils import comparison_metrics

import math
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from torchvision.transforms.v2 import functional as F, InterpolationMode

from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2


def _apply_op(
    im: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        im = F.affine(
            im,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        im = F.rotate(im, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        im = F.adjust_brightness(im, 1.0 + magnitude)
    elif op_name == "Color":
        im = F.adjust_saturation(im, 1.0 + magnitude)
    elif op_name == "Contrast":
        im = F.adjust_contrast(im, 1.0 + magnitude)
    elif op_name == "Sharpness":
        im = F.adjust_sharpness(im, 1.0 + magnitude)
    elif op_name == "Posterize":
        im = F.posterize(im, int(magnitude))
    elif op_name == "Solarize":
        im = F.solarize(im, magnitude)
    elif op_name == "AutoContrast":
        im = F.autocontrast(im)
    elif op_name == "Equalize":
        im = F.equalize(im)
    elif op_name == "Invert":
        im = F.invert(im)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return im


class CustomTrivialAugmentWide(torch.nn.Module):
    def __init__(
        self,
        soft: bool = False,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        severity: int = 0,
        selected_transforms: Optional[List[str]] = None,
        get_signed: bool = False,
        chance: float = 0.1,
        custom_chance: float = 0.7,
        k: int = 2,
        mapping_approach: str = "polynomial_chance",
    ):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        
        """MODIFICATION"""
        self.soft = soft
        self.chance = chance
        self.custom_chance = custom_chance
        self.mapping_approach = mapping_approach
        self.severity = severity
        self.selected_transforms = selected_transforms
        self.get_signed = get_signed
        self.k = k
        """MODIFICATION"""


    def _augmentation_space(self, num_bins: int, selected_transforms: Optional[List[str]] = None, height: int=32, width: int=32) -> Dict[str, Tuple[Tensor, bool]]:
        # Define the full augmentation space
        augmentation_space = {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True), #float(width)
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True), #float(width)
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(1.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

        if selected_transforms is None:
            # Return the full dictionary if no specific transforms are selected
            return augmentation_space
        elif isinstance(selected_transforms, str):
            selected_transforms = [selected_transforms]

        # Validate selected_transforms to ensure they're in the augmentation space
        invalid_transforms = [t for t in selected_transforms if t not in augmentation_space]
        if invalid_transforms:
            raise ValueError(f"Invalid transform names provided: {invalid_transforms}")

        # Return a subset of the augmentation space based on the selected transforms
        return {key: augmentation_space[key] for key in selected_transforms}

    def model_accuracy_mapping(self, augmentation_magnitude: Optional[float], augmentation_type: Optional[str], root_path: Optional[str] = "/kaggle/working/MasterArbeit") -> Optional[float]:
        
        # For local testing
        root_path = f'visualization/non_linear_mapping_data/{augmentation_type}/'
        
        filename = os.path.join(root_path, f"{augmentation_type}_MAPPING_results.csv")
        data = pd.read_csv(filename)
        augmentation_magnitude_list = data["Severity"]
        model_accuracy_list = data["Accuracy"]

        # idx = np.where(augmentation_magnitude_list == augmentation_magnitude)
        for i in range(len(augmentation_magnitude_list)):
            mag = augmentation_magnitude_list[i]
            if round(mag, 5) == round(augmentation_magnitude, 5):
                return model_accuracy_list[i], i
            
    def compute_visibility(self, dim1: int, dim2: int, tx: float, ty: float) -> float:
        """Computes the visibility of the cropped uimage within the background.

        Args:
            dim1 (int): Height of the image.
            dim2 (int): Width of the image.
            tx (int): Horizontal offset.
            ty (int): Vertical offset.

        Returns:
            float: Visibility ratio of the cropped image.
        """
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def apply_standard_augmentation(
        self, im: Tensor, op_meta
    ) -> Tuple[Tensor, Dict[str, float]]:
        fill = self.fill
        channels, height, width = F.get_dimensions(im)

        if isinstance(fill, (int, float)):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]

        """MODIFCATION: Set magnitude and remove signed"""
        if self.severity != -1:
            if magnitudes.ndim > 0:
                magnitude = float(magnitudes[self.severity].item())
            else:
                magnitude = 0.0
        else:
            magnitude = (
                float(
                    magnitudes[
                        torch.randint(len(magnitudes), (1,), dtype=torch.long)
                    ].item()
                )
                if magnitudes.ndim > 0
                else 0.0
            )
        
        if self.get_signed:
            if self.get_signed and op_name not in ["Solarize", "Posterize"]:
                magnitude *= -1.0
        else:
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
        """MODIFICATION: Set magnitude and remove signed"""

        im = _apply_op(
            im, op_name, magnitude, fill=fill, interpolation=self.interpolation
        )

        return im, op_name, magnitude

    def apply_custom_augmentation(self, im: Tensor) -> Tuple[Tensor, List[float]]:
        _ , height, width = F.get_dimensions(im)
        aug_space = self._augmentation_space(self.num_magnitude_bins, selected_transforms=self.selected_transforms, height=height, width=width)
        augment_im, augmentation_type, augmentation_magnitude = self.apply_standard_augmentation(im, aug_space)
        confidence_aa = 1.0  # Default value
        dim1, dim2 = im.shape[1], im.shape[2]

        if self.soft == False:
            # print(f"\nAugmentation info: {augment_info}\tconf: {confidence_aa}\n")
            return augment_im, augmentation_type, augmentation_magnitude, confidence_aa
        
        """Performance data obtained from available HVS"""
        occlusion_hvs = [0.216, 0.388, 0.51066667, 0.584, 0.65333333, 0.68533333, 0.68, 0.72666667, 0.75466667, 0.764, 0.776, 0.78758974, 0.79876923, 0.80994872, 0.82112821, 0.83230769, 0.84348718, 0.85466667, 0.86584615, 0.87702564, 0.88820513, 0.89938462, 0.9105641, 0.92174359, 0.93292308, 0.94410256, 0.95528205, 0.96646154, 0.97764103, 0.98882051, 1.]
        rotation_hvs = [1., 0.9985, 0.997, 0.9955, 0.994, 0.9925, 0.991, 0.9895, 0.988, 0.9865, 0.985, 0.9835, 0.982, 0.9805, 0.979, 0.9775, 0.976, 0.9745, 0.973, 0.9715, 0.97, 0.964, 0.958, 0.952, 0.946, 0.94, 0.934, 0.9315, 0.936, 0.9405, 0.945]
        contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
        """Performance data obtained from available HVS"""

        if augmentation_type in ['Identity', 'AutoContrast', 'Equalize', 'Invert']:
            augmentation_idx = 0
        else:
            mags = aug_space[augmentation_type]
            for i in range(len(mags[0])):
                if round(abs(augmentation_magnitude), 5) == round(mags[0][i].item(), 5):
                    augmentation_idx = i
                    break

        if self.mapping_approach=="ssim_metric":
            """Image Similarity Metric"""
            if augmentation_type=="Contrast":
                confidence_aa = comparison_metrics.multiscale_contrast_similarity(im, augment_im)
            else:
                confidence_aa = comparison_metrics.multiscale_structural_similarity(im, augment_im)
        elif self.mapping_approach=="uiq_metric":
            confidence_aa = comparison_metrics.universal_image_quality_index(im, augment_im)
        elif self.mapping_approach=="ncc_metric":
            confidence_aa = comparison_metrics.normalized_cross_correlation(im, augment_im)
        elif self.mapping_approach=="scc_metric":
            confidence_aa = comparison_metrics.spatial_correlation_coefficient(im, augment_im)
        elif self.mapping_approach=="sift_metric":
            if augmentation_type not in ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]:
                confidence_aa = 1.0
            else:
                confidence_aa = comparison_metrics.sift_correction_factor(im, augment_im)
        else:
            if augmentation_type == "ShearX":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    k = 1.0  # 1.5 1.0
                    chance = 0.9315  # 0.224 0.9315
                    confidence_aa = 1 - (1 - chance) * abs(augmentation_magnitude) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.chance) * abs(augmentation_magnitude) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.custom_chance) * abs(augmentation_magnitude) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Rotation HVS"""
                    confidence_aa = rotation_hvs[augmentation_idx]
                elif self.mapping_approach=="other":
                    confidence_aa = 1 - (1 - self.custom_chance) * abs(augmentation_magnitude) ** self.k
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "ShearY":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    k = 1.0  # 1.5, 1.0
                    chance = 0.9315  # 0.226 0.9315
                    confidence_aa = 1 - (1 - chance) * abs(augmentation_magnitude) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.chance) * abs(augmentation_magnitude) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.custom_chance) * abs(augmentation_magnitude) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Rotation HVS"""
                    confidence_aa = rotation_hvs[augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "TranslateX":  # HVS Available
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Occlusion HVS and model accuracy"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=augmentation_magnitude, ty=0
                    )
                    k = 4               # 2, 4
                    chance = 0.216        # 0.102, 0.216 
                    confidence_aa = 1 - (1 - chance) * (1 - visibility) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=augmentation_magnitude, ty=0
                    )
                    confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=augmentation_magnitude, ty=0
                    )
                    confidence_aa = 1 - (1 - self.custom_chance) * (1 - visibility) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Occlusion HVS"""
                    confidence_aa = occlusion_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "TranslateY":  # HVS Available
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Occlusion HVS and model accuracy"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=0, ty=augmentation_magnitude
                    )
                    k = 4                                   # 2, 4
                    chance = 0.216                          # 0.102, 0.216
                    confidence_aa = 1 - (1 - chance) * (1 - visibility) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=0, ty=augmentation_magnitude
                    )
                    confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    visibility = self.compute_visibility(
                        dim1=dim1, dim2=dim2, tx=0, ty=augmentation_magnitude
                    )
                    confidence_aa = 1 - (1 - self.custom_chance) * (1 - visibility) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Occlusion HVS"""
                    confidence_aa = occlusion_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Brightness":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Contrast HVS"""
                    k_neg, k_pos = 5, 3                     # (3, 2), (20, 3) 
                    chance_pos = 0.86                       # model_acc[-1]
                    chance_neg = 0.32                       # 0.102, 0.32
                    if augmentation_magnitude>0.0:
                        confidence_aa = 1 - (1 - chance_pos) * (augmentation_magnitude) ** k_pos
                    else:
                        confidence_aa = 1 - (1 - chance_neg) * (abs(augmentation_magnitude)) ** k_neg
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k
                    
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude)) ** self.k
                    
                elif self.mapping_approach=="exact_hvs":
                    """Exact Contrast HVS"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = contrast_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Contrast":  # HVS Available
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Contrast HVS"""
                    k_neg, k_pos = 5, 2                # (3, 2), (20, 3) 
                    chance_pos = 0.976                   # model_acc[-1]
                    chance_neg = 0.32                   # 0.102, 0.32
                    if augmentation_magnitude>0.0:
                        confidence_aa = 1 - (1 - chance_pos) * (augmentation_magnitude) ** k_pos
                    else:
                        confidence_aa = 1 - (1 - chance_neg) * (abs(augmentation_magnitude)) ** k_neg
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude)) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Contrast HVS"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = contrast_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Color":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    k = 3                       # 2, 5   
                    chance = 0.95               # 0.1, 0.95   
                    if augmentation_magnitude>0.0:
                        confidence_aa = 1.0
                    else:
                        confidence_aa = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k
                    
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude)) ** self.k
                    
                #elif self.mapping_approach=="exact_hvs":
                #    """Exact Contrast HVS"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                #    confidence_aa = contrast_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Sharpness":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    k_neg = 4                       # 2, 7   
                    k_pos = 1
                    chance_neg = 0.884                 # 0.1, 0.884 
                    chance_pos = 0.992  
                    if augmentation_magnitude>0.0:
                        confidence_aa = 1 - (1 - chance_pos) * (abs(augmentation_magnitude)) ** k_pos
                    else:
                        confidence_aa = 1 - (1 - chance_neg) * (abs(augmentation_magnitude)) ** k_neg
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k
                    
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude)) ** self.k
                    
                #elif self.mapping_approach=="exact_hvs":
                #    """Exact Contrast HVS"""
                    #if augmentation_magnitude>0.0:
                    #    confidence_aa = 1.0
                    #else:
                #    confidence_aa = contrast_hvs[::-1][augmentation_idx]
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Posterize":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    scaled_magnitude = (augmentation_magnitude - 2) / 6
                    k = 10           
                    chance = 0.86      
                    confidence_aa = 1 - (1 - chance) * (1 - scaled_magnitude) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.chance) * (1 - float(augmentation_magnitude / 8.0)) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.custom_chance) * (1 - float(augmentation_magnitude / 8.0)) ** self.k
                else:
                    confidence_aa = 1.0
                
            elif augmentation_type == "Solarize":
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Model Accuracy"""
                    k = 1.5          
                    chance = 0.512
                    confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.chance) * (1 - augmentation_magnitude) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.custom_chance) * (1 - augmentation_magnitude) ** self.k
                elif self.mapping_approach=="other":
                    confidence_aa = 1 - (1 - self.custom_chance) * (1 - augmentation_magnitude) ** self.k
                else:
                    confidence_aa = 1.0

            elif augmentation_type == "Rotate":  # HVS Available
                if self.mapping_approach=="exact_model_accuracy":
                    """Exact Model Accuracy"""
                    confidence_aa, _ = self.model_accuracy_mapping(augmentation_magnitude, augmentation_type)
                elif self.mapping_approach=="smoothened_hvs_or_model_accuracy":
                    """Mapping function from Rotation HVS"""
                    k = 1  # 2, 3
                    chance = 0.9315 # 0.2, 0.9315
                    confidence_aa = 1 - (1 - chance) * (abs(augmentation_magnitude) / 135.0) ** k
                elif self.mapping_approach=="polynomial_chance":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude) / 135.0) ** self.k
                elif self.mapping_approach=="polynomial_custom":
                    """Fixed Parameters"""
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude) / 135.0) ** self.k
                elif self.mapping_approach=="exact_hvs":
                    """Exact Rotation HVS"""
                    confidence_aa = rotation_hvs[augmentation_idx]
                elif self.mapping_approach=="other":
                    confidence_aa = 1 - (1 - self.custom_chance) * (abs(augmentation_magnitude) / 135.0) ** self.k
                else:
                    confidence_aa = 1.0

            #elif augmentation_type in ['AutoContrast', 'Equalize']:
            #    confidence_aa = 0.9

        confidence_aa = torch.from_numpy(
            np.where(confidence_aa < self.chance, self.chance, confidence_aa)
        )

        if isinstance(confidence_aa, torch.Tensor):
            confidence_aa = confidence_aa.item()

        return augment_im, augmentation_type, augmentation_magnitude, confidence_aa
        
    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"soft={self.soft}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )

        return s
    
    def forward(self, im: torch.Tensor) -> Tensor:
        # if self.soft:
        aug_im, aug_type, aug_mag, conf = self.apply_custom_augmentation(im)
        return aug_im, aug_type, aug_mag, conf
