import torch
import math
import numbers
import warnings
from typing import List, Optional, Sequence, Tuple
from torch import Tensor
from torchvision.transforms import functional as F


class RandomErasing(torch.nn.Module):
    """Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0, inplace=False, custom=False, chance: float = 0.1):
        super().__init__()
        # _log_api_usage_once(self)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        
        """MODIFICATION"""
        self.chance = chance
        self.k = 1
        self.custom = custom

        self.occlusion_hvs = [0.216, 0.388, 0.51066667, 0.584, 0.65333333, 0.68533333, 0.68, 0.72666667, 0.75466667, 0.764, 0.776, 0.78758974, 0.79876923, 0.80994872, 0.82112821, 0.83230769, 0.84348718, 0.85466667, 0.86584615, 0.87702564, 0.88820513, 0.89938462, 0.9105641, 0.92174359, 0.93292308, 0.94410256, 0.95528205, 0.96646154, 0.97764103, 0.98882051, 1.]

        """MODIFICATION"""

    @staticmethod
    def get_params(
        img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        # to_tensor = F.to_tensor
        # img = to_tensor(img[0])

        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def compute_visibility(self, dim1: int, dim2: int, h: float, w: float) -> float:
        """Computes the visibility of the cropped uimage within the background.

        Args:
            dim1 (int): Height of the image.
            dim2 (int): Width of the image.
            tx (int): Horizontal offset.
            ty (int): Vertical offset.

        Returns:
            float: Visibility ratio of the cropped image.
        """
        return 1 - (h * w) / (dim1 * dim2)
    
    def ensure_tuple_and_append(self, existing, value):
        return (existing if isinstance(existing, tuple) else (existing,)) + (value,)

    def forward(self, img):
        """
        Args:
            image: tuple containing Tensor image, TA magnitude, confidence.

        Returns:
            tuple containing cropped Tensor image, TA magnitude, confidences tuple.
        """

        """MODIFICATION"""
        h, w = 0, 0

        confidence_re = 1.0

        tensor_image = img[0]
        if not isinstance(tensor_image, torch.Tensor):
            raise TypeError(f"Expected Tensor Image but got {type(tensor_image)}")

        confidences = img[3]

        """MODIFICATION"""

        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and not (len(value) in (1, tensor_image.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{tensor_image.shape[-3]} (number of input channels)"
                )


            x, y, h, w, v = self.get_params(tensor_image, scale=self.scale, ratio=self.ratio, value=value)
            
            if self.custom:
                visibility = self.compute_visibility(tensor_image.shape[-2], tensor_image.shape[-1], h, w)
                confidence_re = (
                    1 - (1 - self.chance) * (1 - visibility) ** self.k
                )  # The non-linear function

            tensor_image = F.erase(tensor_image, x, y, h, w, v, self.inplace)
        
        return tensor_image, img[1], img[2], self.ensure_tuple_and_append(confidences, confidence_re)


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"scale={self.scale}, "
            f"ratio={self.ratio}, "
            f"value={self.value}, "
            f"inplace={self.inplace})"
        )
        return s