from model.model_cifar10 import CIFAR10Classifier, EmbeddingWrapper, CustomTransformerModel
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import pandas as pd

classification_path = "model/cifar10_model.pth"
latent_dim = "256"
device = "cuda"
num_classes = 4

# Load the CIFAR10 model
classifier = CIFAR10Classifier()
classifier.load_state_dict(torch.load(classification_path))
classifier.to(device)
classifier.eval()
embedding_model = classifier.conv_layers

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR10 dataset
cifar_10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar_10_data = cifar_10_dataset.data
cifar_10_labels = cifar_10_dataset.targets

all_embeddings = []

# Iterate over batches of data
for i in tqdm(range(0, len(cifar_10_data), 1000)):
    batch = cifar_10_data[i:i + 1000]
    labels = cifar_10_labels[i:i + 1000]

    batch = torch.tensor(batch).permute(0, 3, 1, 2).float().to(device)
    # Generate embeddings
    embeddings = embedding_model(batch)
    
    # Append embeddig to all_embeddings
    all_embeddings.append(embeddings.cpu().detach().numpy())

# Reshape von (10, 1000, 256) -> (10000, 256)
all_embeddings = np.array(all_embeddings).reshape(-1, int(latent_dim))

# Create pandas dataframe with embeddings and labels
df = pd.DataFrame()
df["label"] = cifar_10_labels
df["embedding"] = all_embeddings.tolist()

# Save dataframe to csv
df.to_csv("data/cifar10_embeddings.csv", index=False)