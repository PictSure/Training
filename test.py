from model.model import Autoencoder, CustomTransformerModel, EncocderWrapper
from utils.data_loader import get_mnist_random_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

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
            images, labels, pred_image, pred_label = images.to(device), labels.to(device), pred_image.to(device), pred_label.to(device)

            outputs = model(images, labels, pred_image)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == pred_label).sum().item()
            total += pred_label.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")
    return accuracy

batch_size = 16
device = "cuda"
latent_dim = 64

autoencoder = Autoencoder(latent_dim).to(device)
encoder = autoencoder.encoder
encoder = EncocderWrapper(encoder)
model = CustomTransformerModel(encoder, 2, device)
model.load_state_dict(torch.load("model/transformer.pth"))

# Usage
loader = get_mnist_random_loader(batch_size=batch_size, train=False)
test_plotter = test_model(model, loader, device)