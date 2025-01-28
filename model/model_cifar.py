import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    A simple 2D ResNet Basic Block:
      - Two consecutive 3x3 convolutions + batch norm + ReLU
      - Skip connection (shortcut) possibly with 1x1 convolution if dimensions differ
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # If there's a change in number of channels or a stride > 1, 
        # we define a 1x1 convolution for the shortcut.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out

class CIFAR100Classifier(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100Classifier, self).__init__()
        
        # 1) Pre-processing layers: 
        #    - Initial conv + batchnorm + ReLU
        #    - Stack of BasicBlocks
        #    - Final global average pooling
        self.pre_processing = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
            
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 2) Final classification layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Pre-processing: feature extraction
        out = self.pre_processing(x)
        # Flatten after the global average pool
        # out = out.view(out.size(0), -1)
        # Classification
        out = self.fc(out)
        return out


# 2. Model Definition
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.latent_dim = 64 * 8 * 8
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class EmbeddingWrapper(nn.Module):
    def __init__(self, classifier):
        super(EmbeddingWrapper, self).__init__()
        self.classifier = classifier
        self.latent_dim = 128
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize using CIFAR-10 stats

    def forward(self, x):
        # (batch, num_images, 3, 32, 32) -> (batch * num_images, 3, 32, 32)
        num_images = x.size(1)
        batch_size = x.size(0)

        x = x.view(-1, 3, 32, 32)
        x = self.normalize(x)
        x = self.classifier.pre_processing(x)

        # (batch * num_images, 256) -> (batch, num_images, 256)
        x = x.view(batch_size, num_images, self.latent_dim)
        return x
    
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
        self.x_projection = nn.Linear(embedding_layer.latent_dim, 256).to(device)
        self.y_projection = nn.Linear(num_classes, 256).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4).to(device)
        self.fc = nn.Linear(512, num_classes).to(device)
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

        # Project y_train (scalar or one-hot) to 32D
        y_train = y_train.unsqueeze(-1) if y_train.ndim == 1 else y_train  # Ensure shape (batch, seq, 1)
        # y_train = y_train.float()

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
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, 32)

        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device) -1  # Shape: (batch, seq, projection_dim)

        # Concatenate x_pred and y_pred projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Concatenate train and prediction embeddings
        # pred_combined_embedded = pred_combined_embedded.unsqueeze(1)

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
        # use mean instead
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        return logits