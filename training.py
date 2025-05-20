import torch
import numpy as np
import argparse
import random
from models.resnets import WideResNet_28_4, ResNet18
from augment_dataset import load_data, create_transforms, load_data_c_separately
from utils.utils import str2bool, seed_worker
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import csv
from itertools import zip_longest

device = "cuda" if torch.cuda.is_available() else "cpu"

def soft_loss(pred, label, confidence, reweight=False):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target with correct label = confidence and the inconfidence uniformly distributed
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1).float()

    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label, src=confidence)
    
    # hard one_hot
#     one_hot = torch.zeros_like(pred)
#     one_hot.scatter_(dim=1, index=label, value=1.0)
    # Compute weighted KL loss
    kl = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    kl = kl.unsqueeze(1)  # Unweighted
    if reweight:
        kl = confidence * kl  # Weighted
    return kl.mean()

if __name__ == "__main__":

    par = argparse.ArgumentParser(description='Soft Augmentations')
    par.add_argument('--seed', default=0, type=int, help='seed number')
    par.add_argument('--batch_size', default=256, type=int, help='train batch size')
    par.add_argument('--selected_transforms', type=str, nargs='+', default=None)
    par.add_argument('--augmentation_sign', type=str2bool, nargs='?', const=False, default=False)
    par.add_argument('--augmentation_severity', default=-1, type=int)
    par.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10, CIFAR100 and TinyImageNet are avaliable.")
    par.add_argument('--random_cropping', default=1, type=int, help='0 for none, 1 for original, 2 for soft')
    par.add_argument('--trivial_augment', default=0, type=int, help='0 for none, 1 for original, 2 for soft')
    par.add_argument('--random_erasing', default=0, type=int, help='0 for none, 1 for original, 2 for soft')
    par.add_argument('--random_erasing_p', default=0.3, type=float, help='random erasing probability')
    par.add_argument('--random_erasing_max_scale', default=0.33, type=float, help='random erasing maximum scale (image area)')
    par.add_argument('--epochs', default=200, type=int, help='number epochs (cosine scheduler)')
    par.add_argument('--learning_rate', default=0.1, type=float, help='initial learning rate (cosine scheduler)')
    par.add_argument('--reweight', type=str2bool, nargs='?', const=False, default=False, help='reweighting loss with confidence')

    args = par.parse_args()

    # Set the random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Mapping approach for confidence calculation
    """
    Mapping Approaches:
    1. exact_model_accuracy
    2. smoothened_hvs
    3. fixed_params
    4. exact_hvs
    5. ssim_metric
    6. uiq_metric
    7. ncc_metric
    8. scc_metric
    9. sift_metric
    """
    # Create the transformations for preprocessing and augmentation
    transforms_preprocess, transforms_augmentation = create_transforms(random_cropping=args.random_cropping,
                                                                       trivial_augment=args.trivial_augment,
                                                                       random_erasing=args.random_erasing,
                                                                       random_erasing_p=args.random_erasing_p,
                                                                       random_erasing_max_scale=args.random_erasing_max_scale,
                                                                       selected_transforms=args.selected_transforms,
                                                                       augmentation_severity=args.augmentation_severity, 
                                                                       augmentation_sign=args.augmentation_sign, 
                                                                       dataset_name=args.dataset,
                                                                       seed=args.seed,
                                                                       mapping_approach="fixed_params")
    
    print(transforms_augmentation)

    # Load the CIFAR-10 dataset with the specified transformations
    trainset, testset, num_classes = load_data(transforms_preprocess=transforms_preprocess, 
                                  transforms_augmentation=transforms_augmentation, 
                                  dataset_name=args.dataset)

    # Create a data loader for the training set
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=0,
                                              worker_init_fn=seed_worker, 
                                              generator=g)
    testloader = torch.utils.data.DataLoader(testset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False, 
                                              num_workers=0)

    net = WideResNet_28_4(num_classes=num_classes)
    net.to(device)

    # Initialize the optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)  # Cosine Annealing LR Scheduler

    # For plotting
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    # Training loop
    print(f'\nStart Training...\n')
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        start_time = time.time()
        
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        total = 0
        correct = 0
        test_loss = 0.0
        
        net.train()
        
        for i, data in enumerate(trainloader):
                
            inputs, labels, combined_confidences, _ = data
            inputs, labels, combined_confidences = inputs.to(device), labels.to(device), combined_confidences.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Convert labels to one-hot encoded vectors
            # labels_one_hot = F.one_hot(labels, num_classes=10).float()
            
            outputs = net(inputs)
            
            loss = soft_loss(pred=outputs, label=labels, confidence=combined_confidences, reweight=args.reweight)
            #loss = criterion(outputs, labels)
                
            # Check for correct training
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError('Loss calculation not correct')
        
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        average_train_loss = running_loss / len(trainloader)
        train_losses.append(average_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        with torch.no_grad():
            net.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # calculate and print average loss for current epoch
        average_test_loss = test_loss / len(testloader)
        test_losses.append(average_test_loss)
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)
        
        print(f'\nEpoch {epoch + 1} - Train Loss: {average_train_loss:.3f} - Train Accuracy: {100 * train_accuracy: .2f} - Test Loss: {average_test_loss:.3f} - Test Accuracy: {100 * test_accuracy: .2f}')    
        
        scheduler.step()
        end_time = time.time()
        print(f'\nProcessing time: {(end_time - start_time): 3f} seconds.')

    print('Finished Training')

    # Save the trained model
    PATH = f'/models/trained/{args.dataset}_rc-{args.random_crop}_ta-{args.trivial_augment}_re-{args.random_erasing}-p-{args.random_erasing_p}-max-{args.random_erasing_max_scale}_reweight-{args.reweight}.pth'
    torch.save(net.state_dict(), PATH)

    print(f'\nModel saved.')

    # Evaluate
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            clean_accuracy = 100 * correct / total
            print(f'Final Test Accuracy: {clean_accuracy:.2f} %')


        c_datasets_dict, corruptions = load_data_c_separately(args.dataset, testset, args.batch_size, transforms_preprocess)
        
        # List to store average accuracies for each corruption dataset
        average_accuracies = []

        # Loop over corruptions, loading and testing all 5 severity levels of each corruption dataset
        for corruption in corruptions:

            c_dataloader = c_datasets_dict[corruption]

            correct = 0
            total = 0

            for images, labels in c_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(f'Robust Accuracy on {corruption} corruption: {accuracy:.2f} %')
                average_accuracies.append(accuracy)

                # Clear GPU memory
                torch.cuda.empty_cache()

        # Calculate and print the average accuracy for the corruption dataset
        average_accuracy = sum(average_accuracies) / len(average_accuracies)
        average_accuracies.append(average_accuracy)
        print(f'Average overall robust accuracy: {average_accuracy:.2f} %')


    csv_path = f'/models/trained/{args.dataset}_rc-{args.random_crop}_ta-{args.trivial_augment}_re-{args.random_erasing}-p-{args.random_erasing_p}-max-{args.random_erasing_max_scale}_reweight-{args.reweight}_metrics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        #first header line
        writer.writerow(['Clean Accuracy', clean_accuracy, 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        
        # build iterators for each column
        epochs_iter      = range(1, args.epochs + 1)
        corruptions_iter = corruptions
        avgacc_iter      = average_accuracies
        train_loss_iter  = train_losses
        train_acc_iter   = train_accuracies
        test_loss_iter   = test_losses
        test_acc_iter    = test_accuracies

        # zip them all, using '' as fill‐value when one runs out
        for row in zip_longest(
            corruptions_iter, avgacc_iter,
            epochs_iter,     train_loss_iter, train_acc_iter,
            test_loss_iter,  test_acc_iter,
            fillvalue=''
        ):
            writer.writerow(row)


