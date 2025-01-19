import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from model.model import Autoencoder
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 1e-3
num_epochs = 10
latent_dim = 64

device = "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def vae_loss_function(reconstructed, original, mu, log_var, kl_weight=0.5):
    # Reconstruction loss
    recon_loss = nn.BCELoss(reduction='sum')(reconstructed, original.view(original.size(0), -1))
    
    # KL divergence
    kl_div = -kl_weight * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_div, recon_loss, kl_div

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_div = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        reconstructed, mu, log_var = model(data)

        loss, recon_loss, kl_div = vae_loss_function(reconstructed, data, mu, log_var, 2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_div += kl_div.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Recon Loss: {epoch_recon_loss/len(train_loader):.4f}, KL Div: {epoch_kl_div/len(train_loader):.4f}")

torch.save(model.state_dict(), "model/autoencoder_mnist.pth")

print("Training complete!")

# Plot the original and reconstructed images
model.eval()
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)
    reconstructed, _, _ = model(data)
    break

plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    plt.subplot(2, 10, i+1)
    plt.imshow(data[i].view(28, 28).cpu().detach().numpy(), cmap='gray')

    # Reconstructed
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].view(28, 28).cpu().detach().numpy(), cmap='gray')

plt.savefig("reconstructed_images.png")