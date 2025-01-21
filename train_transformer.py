from model.model import Autoencoder, CustomTransformerModel, EncocderWrapper
from utils.data_loader import get_mnist_random_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import pandas as pd

autoencoder_path = "model/autoencoder_mnist.pth"
latent_dim = 64
device = "cuda"

autoencoder = Autoencoder(latent_dim).to(device)
autoencoder.load_state_dict(torch.load(autoencoder_path))
autoencoder.to(device)
autoencoder.eval()
encoder = autoencoder.encoder
encoder = EncocderWrapper(encoder)

encoder = encoder.to(device)
model = CustomTransformerModel(encoder, 2, device=device)

criterion = nn.CrossEntropyLoss()

batch_size = 16
loader = get_mnist_random_loader(batch_size=batch_size)

from tqdm import tqdm
import time

def test(lr=1e-3, num_epochs=40, batches_per_epoch=1000, log_step=500):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    start_time = time.time()

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0
        losses = []
        accuracies = []

        for batch_idx, (images, labels, pred_image, pred_label) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
            outputs = model.forward(images, labels, pred_image)
            
            pred_label = pred_label.view(-1)
            loss = criterion(outputs, pred_label)
            
            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == pred_label).sum().item()
                total = pred_label.size(0)
                total_correct += correct
                total_samples += total
                losses.append(loss.item())
                accuracies.append(correct / total)

        total_grad_norm = 0.0
        grad_param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                grad_param_count += 1

        avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0

        avg_loss = total_loss / batches_per_epoch
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}, Avg Gradient Norm: {avg_grad_norm:.6f}")

        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
        print(f"Estimated time left: {remaining_time // 60:.0f} minutes {remaining_time % 60:.0f} seconds")
    return avg_loss, accuracy, losses, accuracies

learning_rates = [1e-4]
num_epochs = 10
batches_per_epoch = 10000

results = {}    
for lr in learning_rates:
    model = CustomTransformerModel(encoder, 2, device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    avg_loss, accuracy, losses, accuracies = test(lr=lr, num_epochs=num_epochs, batches_per_epoch=batches_per_epoch)

smoothed_losses = pd.Series(losses).rolling(window=100).mean()
smoothed_accuracies = pd.Series(accuracies).rolling(window=100).mean()

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
plt.title('Loss and Accuracy per Batch')
plt.grid(True)
plt.savefig("accuracy.pdf")

torch.save(model.state_dict(), "model/transformer.pth")