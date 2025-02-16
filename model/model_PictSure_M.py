from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ResNetWrapper(nn.Module):
    def __init__(self, classifier):
        super(ResNetWrapper, self).__init__()
        self.feature_extractor = nn.Sequential(*list(classifier.children())[:-1], torch.nn.Flatten())
        self.latent_dim = self.feature_extractor(torch.zeros(1, 3, 224, 224)).shape[-1]

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.feature_extractor(x)
        x = x.view(batch_size, num_images, self.latent_dim)
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, embedding_layer, num_classes, device="cpu"):
        super(CustomTransformerModel, self).__init__()
        self.x_projection = nn.Linear(embedding_layer.latent_dim, 1024).to(device)
        self.y_projection = nn.Linear(num_classes, 1024).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=2048, nhead=8, dim_feedforward=2048, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6).to(device)
        self.fc = nn.Linear(2048, num_classes).to(device)
        self.device = device
        self._init_weights()

        self.num_classes = num_classes

        self.embedding = embedding_layer.to(device)

        for param in self.embedding.parameters():
            param.requires_grad = True

        self.x_projection.requires_grad = True
        self.y_projection.requires_grad = True
        self.transformer.requires_grad = True
        self.fc.requires_grad = True

    def _init_weights(self):
        # Loop through all modules in the model
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:  # Apply Xavier only to 2D+ parameters
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)  # Bias is initialized to zero

    def forward(self, x_train, y_train, x_pred, embedd=True):
        if embedd:
            x_embedded = self.embedding(x_train)  # Shape: (batch, seq, embedding_dim)
            # (batch, rgb, seq, dim) -> (batch, 1, rgb, seq, dim)
            x_pred = x_pred.unsqueeze(1)
            x_pred_embedded = self.embedding(x_pred)  # Shape: (batch, seq, embedding_dim)
        else:
            x_embedded = x_train
            x_pred_embedded = x_pred

        x_projected = self.x_projection(x_embedded)  # Shape: (batch, seq, projection_dim)

        # Ensure y_train in the right dimensions
        y_train = y_train.unsqueeze(-1) if y_train.ndim == 1 else y_train  # Ensure shape (batch, seq, 1)

        # One-hot encode y_train (batch_size, num_classes * num_images) -> (batch_size, num_images * num_classes, num_classes)
        y_train = F.one_hot(y_train, num_classes=self.num_classes).float()

        # (batch, seq, num_classes) -> (batch * seq, num_classes)
        y_train = y_train.view(-1, self.num_classes)

        y_projected = self.y_projection(y_train)  # Shape: (batch, seq, projection_dim)

        # Reshape back to (batch, seq, projection_dim)
        y_projected = y_projected.view(x_projected.size(0), x_projected.size(1), -1)

        # Concatenate x and y projections
        combined_embedded = torch.cat([x_projected, y_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Applying the same projection to the prediction
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, projection_dim)

        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device) -1  # Shape: (batch, seq, projection_dim)

        # Concatenate x_pred and y_pred projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Concatenate train and prediction embeddings
        full_sequence = torch.cat([combined_embedded, pred_combined_embedded], dim=1)  # Shape: (batch, seq+pred_seq, d_model)

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
        # Calculate final logits
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        
        return logits