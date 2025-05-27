import sys
import os
import random
import numpy as np

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import (
    plot_mean_std,
    get_mean_std,
    save_to_csv,
    plot_mean_std_from_csv,
)
from wideresnet import WideResNet_28_4
from visualization.visualization_utils import evaluate_model
import torch
import time
from augment_dataset import seed_worker


def get_plot(augmentation_type, model, dataset_split: int = 100, worker_init_fn=None, generator=None, iq_metric='ncc'):
    print(
        f"\n============================ Processing augmentation type: {augmentation_type} ============================\n"
    )
    start_time = time.time()
    mean_list, std_list, time_list, augmentation_magnitudes_list, accuracy_list = [], [], [], [], []

    if augmentation_type in ['Posterize', 'Solarize']:
        sign = 1
    else:
        sign = 2

    for enable_sign in range(0, sign):
        if enable_sign == 1:
            enable_sign = False
        else:
            enable_sign = True

        for severity in range(0, 31):
            total_time = 0
            print(f"Processing severity: {severity} with sign: {enable_sign}\n")
            preprocess, augmentation = create_transforms(
                random_cropping=0,
                trivial_augment=2,
                random_erasing=0,
                random_erasing_p=0.3,
                random_erasing_max_scale=0.33,
                selected_transforms=augmentation_type,
                augmentation_severity=severity, 
                augmentation_sign=enable_sign, 
                dataset_name='CIFAR10',
                mapping_approach="fixed_params"
            )
            
            trainset, _ , _ = load_data(
                transforms_preprocess=preprocess,
                transforms_augmentation=augmentation,
                dataset_name='CIFAR10',
            )

            dataloader = torch.utils.data.DataLoader(
                trainset,
                shuffle=True,
                batch_size=dataset_split,
                num_workers=2,
                worker_init_fn=worker_init_fn,
                generator=generator
            )

            # Confidence Calculation
            _, _, confidences, magnitudes = next(iter(dataloader))
            augmentation_magnitude = magnitudes[0]
            mean, std = get_mean_std(confidences)
            mean_list.append(mean.item())
            std_list.append(std.item())
            augmentation_magnitudes_list.append(augmentation_magnitude)

            # Model Confidence Calculation
            if model is not None:
                accuracy = evaluate_model(model=model, dataloader=dataloader)
                accuracy_list.append(accuracy)
                print(f"Accuracy: {accuracy*100:.2f}%")
            else:
                print(f"\nModel not provided. Skipping model evaluation.\n")

    if len(accuracy_list) == 0:
        print('its going here')
        accuracy_list = [0] * len(mean_list)
    csv_filename = save_to_csv(
        mean_list,
        std_list,
        accuracy_list,
        augmentation_type,
        augmentation_magnitudes_list,
        time_list,
        iq_metric=iq_metric
    )
    print(f'CSV Filename: {csv_filename}')
    plot_mean_std_from_csv(csv_file=csv_filename, augmentation_type=augmentation_type, iq_metric=iq_metric)
    
    end_time = time.time()
    total_time += end_time - start_time

    print(
        f"\n======================= Finished in {total_time:.2f} seconds\n ========================\n"
    )


if __name__ == "__main__":
    augmentation_types = [
        # "Identity",
        # "ShearX",
        # "ShearY",
        # "TranslateX",
        # "TranslateY",
        # "Rotate",
        # "Brightness",
        # "Color",
        "Contrast",
        # "Sharpness",
        # "Posterize",
        # "Solarize",
        # "AutoContrast",
        # "Equalize",
    ]

    g = torch.Generator()
    g.manual_seed(0)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load the saved model weights
    net = WideResNet_28_4(num_classes=10)
    PATH = "models/pretrained/noTA_CIFAR10.pth"
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(PATH, map_location=torch.device("cpu"), weights_only=False)
    net.load_state_dict(state_dict["model_state_dict"], strict=False)

    for augmentation_type in augmentation_types:
        get_plot(augmentation_type, model=net, dataset_split=100, worker_init_fn=seed_worker, generator=g)

        # csv_filename = f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv'
        # iq_metric = 'poly_k'
        # augmentation_type = 'Contrast'

        # print(f'CSV Filename: {csv_filename}')
        # plot_mean_std_from_csv(csv_file=csv_filename, augmentation_type=augmentation_type, iq_metric=iq_metric)
