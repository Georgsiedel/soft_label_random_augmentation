import torch
import numpy as np
import random
import time
import csv
from itertools import zip_longest
from typing import Optional, List
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.resnets import WideResNet_28_4, ResNet18
from augment_dataset import load_data, create_transforms, load_data_c_separately
from utils.train_utils import soft_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(
    seed: int = 0,
    batch_size: int = 256,
    selected_transforms: Optional[List[str]] = None,
    augmentation_sign: bool = False,
    augmentation_severity: int = -1,
    dataset: str = 'CIFAR10',
    random_cropping: int = 1,
    trivial_augment: int = 0,
    random_erasing: int = 0,
    random_erasing_p: float = 0.3,
    random_erasing_max_scale: float = 0.33,
    epochs: int = 2,
    learning_rate: float = 0.1,
    reweight: bool = False,
    mapping_approach: str = "fixed_params",
    save_dir: str = "../trained_models/soft_augmentation",
    results_dir: str = "results"
):
    # 1. Seed everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # 2. Build transforms
    transforms_preprocess, transforms_augmentation = create_transforms(
        random_cropping=random_cropping,
        trivial_augment=trivial_augment,
        random_erasing=random_erasing,
        random_erasing_p=random_erasing_p,
        random_erasing_max_scale=random_erasing_max_scale,
        selected_transforms=selected_transforms,
        augmentation_severity=augmentation_severity,
        augmentation_sign=augmentation_sign,
        dataset_name=dataset,
        seed=seed,
        mapping_approach=mapping_approach
    )

    # 3. Load data
    trainset, testset, num_classes = load_data(
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
        dataset_name=dataset
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=1, worker_init_fn=seed_worker, generator=g
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 4. Model, loss, optimizer, scheduler
    net = WideResNet_28_4(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training loop
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    print(f"\nStarting training: seed={seed}, dataset={dataset}\n")
    steps = ((int(len(trainset) / batch_size) + int(len(testset) / batch_size) + 2) * epochs)
    with tqdm(total=steps, desc="Training Progress") as pbar:

        for epoch in range(epochs):
            start = time.time()
            net.train()
            running_loss, correct_train, total_train = 0., 0, 0

            for inputs, labels, combined_confidences, _ in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                combined_confidences = combined_confidences.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = soft_loss(
                    pred=outputs,
                    label=labels,
                    confidence=combined_confidences,
                    reweight=reweight
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = outputs.max(1)
                total_train += labels.size(0)
                correct_train += (preds == labels).sum().item()

                # Update the progress bar
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "train_acc": f"{correct_train / total_train:.2%}",
                    "correct": f"({correct_train} / {total_train})"
                })
                pbar.update(1)

            train_loss = running_loss / len(trainloader)
            train_acc = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # eval
            net.eval()
            test_loss, correct, total = 0., 0, 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    test_loss += criterion(outputs, labels).item()
                    _, preds = outputs.max(1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                    # Update the progress bar for evaluation
                    pbar.set_postfix({
                        "epoch": f"{epoch+1}/{epochs}",
                        "test_acc": f"{correct / total:.2%}",
                        "correct": f"({correct} / {total})"
                    })
                    pbar.update(1)


            test_loss /= len(testloader)
            test_acc = correct / total
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            scheduler.step()
            elapsed = time.time() - start
            # Update the progress bar with final epoch metrics
            pbar.set_postfix({
                "epoch": f"{epoch+1}/{epochs}",
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2%}",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc:.2%}",
                "time": f"{elapsed:.1f}s"
            })

    # 6. Save model
    fname = (
        f"{dataset}_rc-{random_cropping}"
        f"_ta-{trivial_augment}_re-{random_erasing}"
        f"-p-{random_erasing_p}-max-{random_erasing_max_scale}"
        f"_reweight-{reweight}_seed-{seed}.pth"
    )
    path = f"{save_dir}/{dataset}/{fname}"
    torch.save(net.state_dict(), path)
    print(f"Model saved to {path}")

    # 7. Final clean accuracy
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    clean_acc = 100 * correct / total
    print(f"Final clean accuracy: {clean_acc:.2f}%")

    # 8. Robustness on corruptions
    c_datasets_dict, corruptions = load_data_c_separately(
        dataset, testset, batch_size, transforms_preprocess
    )
    robust_accs = []
    for corr in corruptions:
        corr_loader = c_datasets_dict[corr]
        corr_correct, corr_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in corr_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = net(imgs)
                _, pr = outs.max(1)
                corr_total += lbls.size(0)
                corr_correct += (pr == lbls).sum().item()
        acc = 100 * corr_correct / corr_total
        print(f"Robust acc on {corr}: {acc:.2f}%")
        robust_accs.append(acc)
    avg_robust = sum(robust_accs) / len(robust_accs)
    print(f"Average robust accuracy: {avg_robust:.2f}%")

    # 9. Save CSV metrics
    csv_name = fname.replace(".pth", "_metrics.csv")
    csv_path = f"{results_dir}/{dataset}/{csv_name}"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Clean Accuracy", clean_acc,
            "Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc"
        ])
        for row in zip_longest(
            corruptions, robust_accs,
            range(1, epochs+1), train_losses, train_accuracies,
            test_losses, test_accuracies,
            fillvalue=""
        ):
            w.writerow(row)
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    train()