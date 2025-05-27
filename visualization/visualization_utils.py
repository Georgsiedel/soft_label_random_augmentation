import torch
import os
import pandas as pd

def plot_severity_vs_confidence(augmentation_type: str, data_cat: str = "mean"):
    
    filename = os.path.join(f"./visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_comparison_all_results.csv")
    df = pd.read_csv(filename)
    df = df.sort_values("severity")
    ssim_values = df[f"{data_cat}_ssim"]
    ncc_values = df[f"{data_cat}_ncc"]
    uiq_values = df[f"{data_cat}_uiq"]
    scc_values = df[f"{data_cat}_scc"]

    if augmentation_type in ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]:
        sift_values = df[f"{data_cat}_sift"]
        return ssim_values, ncc_values, uiq_values, scc_values, sift_values
    return ssim_values, ncc_values, uiq_values, scc_values

def evaluate_model(model, dataloader):
    # Evaluate the model
    correct, total = 0, 0

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(dataloader):
            print(i)
            images, labels, _ , _ = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
        # print(
        #     f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
        # )
        return accuracy


def compute_occlusion_visibility(dim1: int, dim2: int, tx: float, ty: float) -> float:
    #see random crop augmentation from soft augmentation
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