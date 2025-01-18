# Define a Custom Transformer Model Using the Pretrained Embedding Layer
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)  # Mean of the latent distribution
        self.fc_log_var = nn.Linear(128, latent_dim)  # Log variance of the latent distribution
        self.relu = nn.ReLU()
        self.latent_dim = latent_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.sigmoid(self.fc3(z))
        return z


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample epsilon from standard normal
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var
    
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

        mu, log_var = self.encoder(x)

        # Return the to shape (batch, num_images, latent_dim)
        x = mu.view(batch_size, num_images, self.encoder.latent_dim)
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, embedding_layer, num_classes, device="cpu"):
        super(CustomTransformerModel, self).__init__()
        self.embedding = embedding_layer.to(device)

        self.x_projection = nn.Linear(embedding_layer.latent_dim, 32).to(device)
        self.y_projection = nn.Linear(1, 32).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6).to(device)
        self.fc = nn.Linear(64, num_classes).to(device)
        self.device = device

        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x_train, y_train, x_pred):
        # Get embeddings for x_train and project to 32D
        x_embedded = self.embedding(x_train)  # Shape: (batch, seq, latent_dim)
        x_projected = self.x_projection(x_embedded)  # Shape: (batch, seq, 32)

        # Project y_train (scalar or one-hot) to 32D
        y_train = y_train.unsqueeze(-1) if y_train.ndim == 1 else y_train  # Ensure shape (batch, seq, 1)
        # Make sure y_train is a float tensor
        y_train = y_train.float()

        # (batch, seq, 1) -> (batch * seq, 1)
        y_train = y_train.view(-1, 1)
        y_projected = self.y_projection(y_train)  # Shape: (batch, seq, 32)
        # Reshape back to (batch, seq, 32)
        y_projected = y_projected.view(x_projected.size(0), x_projected.size(1), -1)

        # Concatenate x and y projections
        combined_embedded = torch.cat([x_projected, y_projected], dim=-1)  # Shape: (batch, seq, 64)

        # Handle x_pred: Embed and project, but use zero for y_pred
        x_pred_embedded = self.embedding(x_pred)  # Shape: (batch, seq, latent_dim)
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, 32)
        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device)  # Shape: (batch, seq, 32)

        # Concatenate x_pred and y_pred projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)  # Shape: (batch, seq, 64)

        # Concatenate train and prediction embeddings
        full_sequence = torch.cat([combined_embedded, pred_combined_embedded], dim=1)  # Shape: (batch, seq+pred_seq, 64)

        # (batch, seq, dim -> seq, batch, dim)
        full_sequence = full_sequence.permute(1, 0, 2)

        # Create an attention mask
        seq_length = full_sequence.size(0)
        attention_mask = torch.ones(seq_length, seq_length, device=self.device)
        attention_mask[-1, :] = 1
        attention_mask[:-1, -1] = 0
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))

        # Pass through transformer encoder
        transformer_output = self.transformer(full_sequence, mask=attention_mask)

        # Extract the prediction hidden state and compute logits
        prediction_hidden_state = transformer_output[-1, :, :]  # Shape: (batch_size, hidden_dim)
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        return logits