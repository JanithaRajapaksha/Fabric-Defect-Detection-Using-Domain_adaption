from __future__ import division
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable

import models
import utils
from data_loader import get_train_test_loader, get_office31_dataloader, get_fabric_dataloader
from main import source_loader, target_loader

from sklearn.metrics import confusion_matrix, classification_report

CUDA = True if torch.cuda.is_available() else False

import seaborn as sns
import matplotlib.pyplot as plt


def test(model, dataset_loader, e):
    model.eval()
    test_loss = 0
    correct = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():  # Prevent gradient computation
        for data, target in dataset_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()

            # Model forward pass
            out, _ = model(data, data)

            # Sum up batch loss
            test_loss += torch.nn.functional.cross_entropy(out, target, reduction='sum').item()

            # Get the index of the max log-probability
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            # Save true and predicted labels for confusion matrix
            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

    test_loss /= len(dataset_loader.dataset)

    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_names = list(target_loader.dataset.class_to_idx.keys())

    plot_confusion_matrix(conf_matrix, classes=class_names)

    # Classification report
    report = classification_report(true_labels, predicted_labels,
                                   target_names=class_names)
    print("Classification Report:")
    print(report)

    return {
        'epoch': e,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * correct / len(dataset_loader.dataset)
    }


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix using matplotlib and seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()
if __name__ == '__main__':
    # Load model
    model = models.DeepCORAL(4)
    if CUDA:
        model = model.cuda()

    utils.load_net(model, 'best_model_checkpoint.tar')

    # Set up data loader (replace `YourDataset` with your dataset class)
    # Example:
    # test_dataset = YourDataset(...)
    # dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Run the test function
    test_results = test(model, target_loader, 0)  # Pass epoch number (e.g., 0 if only testing)
    print(f"Test Results: {test_results}")
