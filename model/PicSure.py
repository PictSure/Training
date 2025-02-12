import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import download_model
import numpy as np

MODEL_URL = "https://cloud.dfki.de/owncloud/index.php/s/tkrXfwc3Q8DjXg4/download"
    
class ResNetWrapper(nn.Module):
    def __init__(self, classifier=None):
        super(ResNetWrapper, self).__init__()
        if classifier is None:
            classifier = models.resnet34(pretrained=True)
        else:
            classifier = classifier
        self.feature_extractor = nn.Sequential(*list(classifier.children())[:-1], torch.nn.Flatten())
        self.latent_dim = self.feature_extractor(torch.zeros(1, 3, 224, 224)).shape[-1]

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.feature_extractor(x)
        x = x.view(batch_size, num_images, self.latent_dim)
        return x
    
    def embedd(self, x):
        x = self.feature_extractor(x)
        return x

class PicSureS(nn.Module):
    def __init__(self, num_classes, embedding_layer=ResNetWrapper(), device="cpu", download=False):
        super(PicSureS, self).__init__()
        if download:
            download_model("model/PicSureModel.pth", MODEL_URL)
            # self.load_state_dict(torch.load("model/PicSureModel.pth"))
        self.x_projection = nn.Linear(embedding_layer.latent_dim, 512)
        self.y_projection = nn.Linear(num_classes, 512)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, dim_feedforward=2048, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(1024, num_classes)
        self.device = device
        self._init_weights()

        self.num_classes = num_classes

        self.context_images = {}

        self.embedding = embedding_layer

    def _init_weights(self):
        # Loop through all modules in the model
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:  # Apply Xavier only to 2D+ parameters
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)  # Bias is initialized to zero

    def setContextImages(self, context_images):
        self.context_images = context_images
        # Check if number of classes in dict {class: [images]} is equal to num_classes
        assert len(self.context_images) == self.num_classes

        # Normalize images
        for class_name, images in self.context_images.items():
            # Check if image is different size than 224x224
            if isinstance(images, list):
                images = torch.stack(images)
                images = torch.tensor(images)
            if images.shape[1:] != (224, 224, 3):
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                self.context_images[class_name] = images
            if images.max() > 1:
                self.context_images[class_name] = images / 255.0
        
        # Convert each image to a tensor
        for class_name, images in self.context_images.items():
            self.context_images[class_name] = torch.tensor(images)

        # Calculate embeddings for each image
        for class_name, images in self.context_images.items():
            self.context_images[class_name] = self.embedding.embedd(images)

        # Apply projections to each embedding
        for class_name, embeddings in self.context_images.items():
            self.context_images[class_name] = self.x_projection(embeddings)

        # One hot encode the class and store images and classes in a list of tuples (image, class)
        context_data = []
        for class_idx, (class_name, embeddings) in enumerate(self.context_images.items()):
            class_one_hot = F.one_hot(torch.tensor([class_idx] * embeddings.size(0)), num_classes=self.num_classes).float().to(embeddings.device)
            class_projections = self.y_projection(class_one_hot)
            context_data.extend([(embedding, class_projections[i]) for i, embedding in enumerate(embeddings)])

        # Create dictionary of class index and class names like {0: 'class_name'}
        self.class_to_idx = {idx: class_name for idx, class_name in enumerate(self.context_images.keys())}

        # Use projections for each class
        self.context_data = context_data

        # Concatenate the projections like in self.forward
        self.context_embeddings = torch.stack([torch.cat([img, cls], dim=-1) for img, cls in self.context_data])

    def predict(self, x_pred):

        # Assume that it is a single image, do preprocessing before embedding
        if x_pred.dim() == 3:  # If the input is a single image (C, H, W)
            x_pred = x_pred.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Ensure the resolution is 224x224
        if x_pred.size(2) != 224 or x_pred.size(3) != 224:
            x_pred = F.interpolate(x_pred, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize the image if necessary
        if x_pred.max() > 1:
            x_pred = x_pred / 255.0

        # Embed the image
        x_pred_embedded = self.embedding(x_pred)  # Shape: (batch, seq, embedding_dim)

        # Project the embedding
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, projection_dim)

        # Create zero projections for the class to be predicted
        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device)  # Shape: (batch, seq, projection_dim)

        # Concatenate the projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Concatenate the context embeddings with the prediction embedding
        full_sequence = torch.cat([self.context_embeddings, pred_combined_embedded], dim=0)  # Shape: (context_seq + pred_seq, d_model)

        # Create transformer mask
        seq_length = full_sequence.size(0)
        attention_mask = torch.ones(seq_length, seq_length, device=self.device)
        attention_mask[-1, :] = 1
        attention_mask[:-1, -1] = 0
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))

        # Pass through transformer
        full_sequence = full_sequence.unsqueeze(1)  # Add batch dimension (1, seq, d_model)
        transformer_output = self.transformer(full_sequence, mask=attention_mask)

        # Extract the prediction hidden state and compute logits
        prediction_hidden_state = transformer_output[-1, :, :]  # Shape: (batch_size, hidden_dim)
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)

        # Apply softmax to logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Return class with highest probability
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return self.class_to_idx[predicted_class]

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

        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device)# Shape: (batch, seq, projection_dim)

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