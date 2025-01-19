import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 2. Model Definition
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x8x8
            nn.Flatten(),  # Flatten the output of conv layers
            nn.Linear(64 * 8 * 8, 256),  # Fully connected layer
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, 10)  # Output layer (10 classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class EmbeddingWrapper(nn.Module):
    def __init__(self, classifier):
        super(EmbeddingWrapper, self).__init__()
        self.classifier = classifier
        self.latent_dim = 256
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize using CIFAR-10 stats

    def forward(self, x):
        # (batch, num_images, 3, 32, 32) -> (batch * num_images, 3, 32, 32)
        num_images = x.size(1)
        batch_size = x.size(0)

        x = x.view(-1, 3, 32, 32)
        x = self.normalize(x)
        x = self.classifier.conv_layers(x)

        # (batch * num_images, 256) -> (batch, num_images, 256)
        x = x.view(batch_size, num_images, 256)
        return x


class CustomTransformerModel(nn.Module):
    def __init__(self, embedding_layer, num_classes, device="cpu"):
        super(CustomTransformerModel, self).__init__()
        self.embedding = embedding_layer.to(device)
        self.x_projection = nn.Linear(embedding_layer.latent_dim, 128).to(device)
        self.y_projection = nn.Linear(1, 128).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6).to(device)
        self.fc = nn.Linear(256, num_classes).to(device)
        self.device = device

        for param in self.embedding.parameters():
            param.requires_grad = False

        self.x_projection.requires_grad = True
        self.y_projection.requires_grad = True
        self.transformer.requires_grad = True
        self.fc.requires_grad = True

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
        # (batch, rgb, seq, dim) -> (batch, 1, rgb, seq, dim)
        x_pred = x_pred.unsqueeze(1)
        x_pred_embedded = self.embedding(x_pred)  # Shape: (batch, seq, latent_dim)
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, 32)
        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device) -1  # Shape: (batch, seq, 32)

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
        # use mean instead
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        return logits