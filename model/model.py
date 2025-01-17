# Define a Custom Transformer Model Using the Pretrained Embedding Layer
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # Keeps embeddings between 0 and 1
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return z

# Decoder Model
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
class EncocderWrapper(nn.Module):
    def __init__(self, encoder):
        super(EncocderWrapper, self).__init__()
        self.encoder = encoder
        self.latent_dim = encoder.latent_dim

    def forward(self, x):
        # Incoming shape (batch, num_images, 1, 28, 28)
        batch_size, num_images, _, _, _ = x.shape

        # Rezhape to (batch * num_images, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)

        x = self.encoder(x)
        # Return the to shape (batch, num_images, latent_dim)
        x = x.view(batch_size, num_images, self.encoder.latent_dim)
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, embedding_layer, num_classes):
        super(CustomTransformerModel, self).__init__()
        self.embedding = embedding_layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_layer.latent_dim + 1, nhead=8
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(embedding_layer.latent_dim + 1, num_classes)

    def forward(self, x_train, y_train):
        # Get embeddings for input tokens
        embedded = self.embedding(x_train)
        # Append the class label to the embeddings
        class_label = torch.zeros_like(embedded[:, :, :1])
        class_label[:, :, 0] = y_train
        embedded = torch.cat([embedded, class_label], dim=-1)

        # (batch, seq, dim -> seq, batch, dim)
        embedded = embedded.permute(1, 0, 2)
        # Pass through transformer encoder
        transformer_output = self.transformer(embedded)

        pooled_output = transformer_output.mean(dim=0)
        logits = self.fc(pooled_output)
        return logits