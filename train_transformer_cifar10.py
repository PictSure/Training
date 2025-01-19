from model.model_cifar10 import CIFAR10Classifier, EmbeddingWrapper, CustomTransformerModel
from utils.data_loader_cifar10 import get_cifar10_random_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

classifier_path = "model/cifar10_model.pth"
latent_dim = 256
device = "cuda"
num_classes = 4

classifier = CIFAR10Classifier()
classifier.load_state_dict(torch.load(classifier_path))
classifier.to(device)
classifier.eval()

encoder = EmbeddingWrapper(classifier)

criterion = nn.CrossEntropyLoss()

batch_size = 16

print("Torch precision: ", torch.get_default_dtype())

def train(lr=1e-3, num_epochs=40, num_images=10):
    initial_lr = lr * 0.01  # Start with a smaller learning rate
    model = CustomTransformerModel(encoder, num_classes, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    start_time = time.time()

    losses = []
    accuracies = []

    loader = get_cifar10_random_loader(batch_size=batch_size, num_classes=num_classes, num_samples=100000, num_images=num_images)

    for epoch in range(num_epochs):
        # Linear ramp up of learning rate only over the first 10 epochs
        if epoch < 10:
            current_lr = initial_lr + (lr - initial_lr) * (epoch / 10)
        else:
            current_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        total_correct = 0
        total_samples = 0
        total_loss = 0

        for batch_idx, (images, labels, pred_image, pred_label) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
            outputs = model.forward(images, labels, pred_image)
            
            pred_label = pred_label.view(-1)
            loss = criterion(outputs, pred_label)
            
            optimizer.zero_grad()   # Reset gradient
            loss.backward()         # Backpropagation

            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()        # Update parameters

            total_loss += loss.item()

            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == pred_label).sum().item()
                total = pred_label.size(0)
                total_correct += correct
                total_samples += total

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        losses.append(avg_loss)
        accuracies.append(accuracy)

        total_grad_norm = 0.0
        grad_param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                grad_param_count += 1

        avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}, Avg Gradient Norm: {avg_grad_norm:.6f}")

        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
        print(f"Estimated time left: {remaining_time // 60:.0f} minutes {remaining_time % 60:.0f} seconds")

    # Save the model
    torch.save(model.state_dict(), f"model/transformer_cifar_{num_images}.pth")
    return avg_loss, accuracy, losses, accuracies


learning_rates = [1e-4]
num_epochs = 30
batches_per_epoch = 10000
 
for lr in learning_rates:
    for num_images in [10]:
        avg_loss, accuracy, losses, accuracies = train(lr=lr, num_epochs=num_epochs)

        smoothed_losses = pd.Series(losses).rolling(window=1).mean()
        smoothed_accuracies = pd.Series(accuracies).rolling(window=1).mean()

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(smoothed_losses, label='Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:orange')
        ax2.plot(smoothed_accuracies, label='Accuracy', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.tight_layout()
        plt.title('Loss and Accuracy per Epoch')
        plt.grid(True)
        plt.savefig(f"accuracy_cifar_{num_images}.pdf")
        plt.show()