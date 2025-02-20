import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def sift_operation(im1, im2, display_matches: bool = False):
    """
    Performs SIFT (Scale-Invariant Feature Transform) on two images and finds matching keypoints.

    Args:
        im1 (PIL.Image or torch.Tensor): First input image.
        im2 (PIL.Image or torch.Tensor): Second input image.
        display_matches (bool, optional): Whether to display the matched keypoints. Defaults to False.

    Returns:
        int: Number of matching keypoints found between the two images.
    """
    if not isinstance(im1, Image.Image) or not isinstance(im2, Image.Image):
        pil = transforms.ToPILImage()
        im1 = pil(im1)
        im2 = pil(im2)
    im1 = im1.convert("L")
    im2 = im2.convert("L")
    im1_np = np.array(im1)
    im2_np = np.array(im2)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1_np, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2_np, None)

    if descriptors1 is None or descriptors2 is None:
        print(
            "Either the images are too different or lacking sufficient features for SIFT to detect"
        )
        return 1

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # print(f"Num matches: {len(matches)}")

    if display_matches:
        im3 = cv2.drawMatches(
            im1_np, keypoints1, im2_np, keypoints2, matches[:500], im2_np, flags=2
        )
        plt.imshow(im3)
        plt.imsave(
            "/home/ekagra/Documents/GitHub/MasterArbeit/example/sift_test_example.png",
            im3,
        )
        plt.title("Number of Matching Keypoints: " + str(len(matches)))
        plt.show()
    # print(f'Number of Matching Keypoints Between the Traning and Query Images: {len(matches)}')
    return len(matches)


