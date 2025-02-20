import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
    
class ProjectionWrapper(nn.Module):
    def __init__(self, input_shape=(16, 16, 3), embedding_dim=256):
        super(ProjectionWrapper, self).__init__()
        self.latent_dim = embedding_dim
        self.input_shape = input_shape
        
        # Projection layers containing one CNN layer and one dense layer
        self.conv_layers = nn.Sequential(
            nn.Flatten(),  # Output: (batch, 16 * 16 * 3)
            nn.Linear(16 * 16 * 3, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Expected input shape: (batch, num_images, 3, 16, 16)
        batch_size, num_images, channels, height, width = x.shape

        # Reshape to process through CNN: (batch * num_images, 3, 16, 16)
        x = x.view(-1, channels, height, width)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Reshape back to (batch, num_images, embedding_dim)
        x = x.view(batch_size, num_images, self.latent_dim)
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, embedding_model, num_classes, num_images=5, patch_size=16, resolution=(224, 224), device="cpu"):
        super(CustomTransformerModel, self).__init__()
        self.y_projection = nn.Linear(num_classes, embedding_model.latent_dim).to(device)
        self.new_token_projection = nn.Linear(1, embedding_model.latent_dim).to(device)

        max_seq_length = ((num_images * num_classes) + 1) * ((resolution[0] // patch_size) * (resolution[1] // patch_size) + 2)

        self.positional_embedding = nn.Embedding(max_seq_length, embedding_model.latent_dim).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_model.latent_dim, nhead=8, dim_feedforward=2048, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4).to(device)
        self.fc = nn.Linear(embedding_model.latent_dim, num_classes).to(device)
        self.device = device
        self._init_weights()

        self.num_classes = num_classes

        self.embedding = embedding_model.to(device)

    def _init_weights(self):
        # Loop through all modules in the model
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:  # Apply Xavier only to 2D+ parameters
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)  # Bias is initialized to zero


    def embedd_images(self, image_data, patch_size=16):

        image_data = image_data.view(-1, *image_data.shape[2:])

        num_patches = image_data.shape[-1] // patch_size
        
        # Split up all images in x_train in patches of 3x16x16
        image_data = image_data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        # Reshape from (batch * num_images, 3, num_patches, num_patches, 16, 16) to (batch * num_images, num_patches x num_patches, 3, 16, 16)
        image_data = image_data.permute(0, 2, 3, 1, 4, 5).reshape(-1, num_patches * num_patches, 3, patch_size, patch_size)

        # Pass through the embedding model
        image_data = self.embedding(image_data)

        return image_data
    
    def create_attention_mask(self, sequence):
        seq_length = sequence.size(0)
        attention_mask = torch.ones(seq_length, seq_length, device=self.device)
        attention_mask[-1, :] = 1
        attention_mask[:-1, -1] = 0
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
        return attention_mask

    def forward(self, x_train, y_train, x_pred, embedd=True):
        
        # x_train_shape: (batch, num_images, 3, 224, 224)
        original_batch_size = x_train.shape[0]

        x_train = x_train.view(-1, *x_train.shape[2:])
    
        patch_size = 16
        batch_size = x_train.shape[0]

        num_patches = x_train.shape[-1] // patch_size
        
        # Create patches for x_train and embedd them
        x_train = x_train.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x_train = x_train.permute(0, 2, 3, 1, 4, 5).reshape(-1, num_patches * num_patches, 3, patch_size, patch_size)
        x_train = self.embedding(x_train)

        # Create patches for x_pred and embedd them
        x_pred = x_pred.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x_pred = x_pred.permute(0, 2, 3, 1, 4, 5).reshape(-1, num_patches * num_patches, 3, patch_size, patch_size)
        x_pred = self.embedding(x_pred)

        # Ensure y_train in the right dimensions
        y_train = y_train.unsqueeze(-1) if y_train.ndim == 1 else y_train  # Ensure shape (batch, seq, 1)

        # One-hot encode y_train (batch_size, num_classes * num_images) -> (batch_size, num_images * num_classes, num_classes)
        y_train = F.one_hot(y_train, num_classes=self.num_classes).float()

        # (batch, seq, num_classes) -> (batch * seq, num_classes)
        y_train = y_train.view(-1, self.num_classes)

        y_projected = self.y_projection(y_train)  # Shape: (batch, seq, projection_dim)

        # Reshape x_train to (batch, seq, num_patches * num_patches, embedding_dim)
        x_train = x_train.view(original_batch_size, -1, num_patches * num_patches, self.embedding.latent_dim)

        # Reshape y_projected to (batch, seq, embedding_dim)
        y_projected = y_projected.view(original_batch_size, -1, self.embedding.latent_dim)

        y_projected = y_projected.unsqueeze(2) # Shape: (batch, seq, 1, embedding_dim)

        # Concatenate y_projected with x_train
        full_context = torch.cat([y_projected, x_train], dim=2)

        y_pred = torch.zeros((x_pred.shape[0], 1, self.embedding.latent_dim), device=self.device)  # Shape: (1, batch, latent_dim)


        full_prediction = torch.cat([y_pred, x_pred], dim=1)

        full_prediction = full_prediction.unsqueeze(1)  # Shape: (batch, seq, 1, embedding_dim)

        # Concatenate full_context with full_prediction
        full_sequence = torch.cat([full_context, full_prediction], dim=1)

        new_tokens = torch.ones((full_sequence.shape[0], full_sequence.shape[1], 1), device=self.device)  # Shape: (batch, seq, 1)

        # Reshape (batch, seq, 1) -> (batch * seq, 1)
        new_tokens = new_tokens.view(-1, 1)

        new_tokens = self.new_token_projection(new_tokens)  # Shape: (batch * seq, embedding_dim)

        # Reshape (batch * seq, embedding_dim) -> (batch, seq, embedding_dim)
        new_tokens = new_tokens.view(original_batch_size, -1, self.embedding.latent_dim)

        # Reshape to (batch, seq, 1, embedding_dim)
        new_tokens = new_tokens.unsqueeze(2)

        # Concatenate full_sequence with new_tokens

        full_sequence = torch.cat([new_tokens, full_sequence], dim=2)

        # Reshape from (batch, seq, num_patches * num_patches, embedding_dim) -> (seq, batch, embedding_dim)
        full_sequence = full_sequence.view(full_sequence.shape[0], full_sequence.shape[1] * full_sequence.shape[2], full_sequence.shape[3])

        # (batch, seq, dim -> seq, batch, dim)
        full_sequence = full_sequence.permute(1, 0, 2)

        # Adding positional encoding
        seq_length = full_sequence.size(0)

        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(1)
        position_embedding = self.positional_embedding(position_ids)

        full_sequence = full_sequence + position_embedding

        attention_mask = self.create_attention_mask(full_sequence)

        # Pass through transformer encoder
        transformer_output = self.transformer(full_sequence, mask=attention_mask)

        # Extract the prediction hidden state and compute logits
        prediction_hidden_state = transformer_output[-1, :, :]  # Shape: (batch_size, hidden_dim)
        # Calculate final logits
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        
        return logits