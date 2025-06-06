import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)

import math
import random
import torch

#torch implementation from https://github.com/gatheluck/PatchGaussian/blob/master/patch_gaussian.py

class AddPatchGaussian():
    def __init__(self, 
                 patch_size: int, 
                 max_scale: float, 
                 randomize_patch_size: bool, 
                 randomize_scale: bool, 
                 custom: bool, 
                 chance: float = 0.3, 
                 k: int = 2, 
                 **kwargs):
        """
        Args:
        - patch_size: size of patch. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert (patch_size >= 1) or (patch_size == -1)
        assert 0.0 <= max_scale <= 1.0

        self.patch_size = patch_size
        self.max_scale = max_scale
        self.randomize_patch_size = randomize_patch_size
        self.randomize_scale = randomize_scale
        self.custom = custom
        self.k = k
        self.chance = chance 
        #Chance being 0.3 and k=2 as standard is based on an approximation from this paper: 
        # https://dl.acm.org/doi/abs/10.1145/3306241
        # where this nonlinear behavior could be estimated

    def __call__(self, img):

        x = img[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected Tensor Image but got {type(x)}")
        confidences = img[3]

        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        assert h == w

        # randomize scale and patch_size
        scale = random.uniform(0, 1) * self.max_scale if self.randomize_scale else self.max_scale
        patch_size = random.randrange(1, self.patch_size + 1) if self.randomize_patch_size else self.patch_size

        gaussian = torch.normal(mean=0.0, std=scale, size=(c, w, h))
        gaussian_image = torch.clamp(x + gaussian, 0.0, 1.0)

        mask = self._get_patch_mask(w, patch_size).repeat(c, 1, 1)

        confidence_gaus = 1.0

        if self.custom:
            #calculate ratio of non-augmented image
            ratio_masked = (mask).float().mean().item()
            confidence_gaus = (
                        1 - (1 - self.chance) * (ratio_masked * scale) ** self.k
                    )  # The non-linear function, with visibility equal to scale of gaussian noise times area covered by it

        patch_gaussian = torch.where(mask == True, gaussian_image, x)

        return patch_gaussian, img[1], img[2], self.ensure_tuple_and_append(confidences, confidence_gaus)

    def _get_patch_mask(self, im_size: int, window_size: int):
        """
        Args:
        - im_size: size of image
        - window_size: size of window. if -1, return full size mask
        """
        assert im_size >= 1
        assert (1 <= window_size) or (window_size == -1)

        # if window_size == -1, return all True mask.
        if window_size == -1:
            return torch.ones(im_size, im_size, dtype=torch.bool)

        mask = torch.zeros(im_size, im_size, dtype=torch.bool)  # all elements are False

        # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
        window_center_h = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)
        window_center_w = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)

        for idx_h in range(window_size):
            for idx_w in range(window_size):
                h = window_center_h - math.floor(window_size / 2) + idx_h
                w = window_center_w - math.floor(window_size / 2) + idx_w

                if (0 <= h < im_size) and (0 <= w < im_size):
                    mask[h, w] = True

        return mask
    
    def ensure_tuple_and_append(self, existing, value):
        return (existing if isinstance(existing, tuple) else (existing,)) + (value,)
