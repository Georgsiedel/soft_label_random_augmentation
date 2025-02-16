import torch

from augment_dataset import create_transforms, load_data


def evaluate_model(model, dataloader):
    # Evaluate the model
    correct, total = 0, 0

    with torch.no_grad():
        model.eval()
        for _, data in enumerate(dataloader):
            images, labels, _ = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
        # print(
        #     f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
        # )
        return accuracy
