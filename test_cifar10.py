from model.model_cifar import CustomTransformerModel, EmbeddingWrapper, CIFAR10Classifier
from utils.data_loader_cifar10 import get_cifar10_random_loader, normalize_samples
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_model(model, test_loader, device):
    """
    Tests the given model on the provided test_loader.

    Args:
        model (torch.nn.Module): The trained model to test.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the test on (CPU or GPU).

    Returns:
        float: Overall accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, pred_image, pred_label in tqdm(test_loader, desc="Testing"):
            # Normalize the images
            images, labels, pred_image, pred_label = images.to(device), labels.to(device), pred_image.to(device), pred_label.to(device)

            outputs = model(images, labels, pred_image)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == pred_label).sum().item()
            total += pred_label.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")
    return accuracy

def plot_sample_predictions(model, test_loader, device):
    """
    Plots the pred_image, the predicted label and the true label.
    """
    model.eval()
    images, labels, pred_image, pred_label = next(iter(test_loader))

    # normalize the images
    images, pred_image = normalize_samples(images, pred_image)
    print(images.shape, pred_image.shape)
    images_norm, labels, pred_image_norm, pred_label = images.to(device), labels.to(device), pred_image.to(device), pred_label.to(device)

    with torch.no_grad():
        outputs = model(images_norm, labels, pred_image_norm)
        predicted = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 2))
    # Reshape from (batch_size, 3, 32, 32) to (batch_size, 32, 32, 3)
    pred_image = pred_image.permute(0, 2, 3, 1).type(torch.int32)
    # pred_image = (pred_image * 0.2023) + 0.4914  # Unnormalize the image

    for i in range(batch_size):
        axes[i, 0].imshow(pred_image[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f"True Label: {pred_label[i].item()}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_image[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Predicted: {predicted[i].item()}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("sample_predictions.png")

batch_size = 16
device = "cuda"

classifier = CIFAR10Classifier().to(device)
encoder = EmbeddingWrapper(classifier)
model = CustomTransformerModel(encoder, 4, device)
model.load_state_dict(torch.load("model/transformer_cifar_10.pth"))

# Usage
loader = get_cifar10_random_loader(batch_size=batch_size, train=False, num_classes=4)
test_plotter = test_model(model, loader, device)

# Plot the sampled images and the predicted image
plot_sample_predictions(model, loader, device)
