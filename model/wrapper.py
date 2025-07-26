import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
    
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
    
class DINOV2Wrapper(nn.Module):
    def __init__(self, device="cpu"):
        super(DINOV2Wrapper, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.device = device
        # Get latent dim by running a dummy input through the model
        dummy = torch.zeros(1, 3, 224, 224)
        inputs = self.processor(images=[dummy.squeeze(0).permute(1,2,0).numpy()], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.latent_dim = outputs.last_hidden_state.shape[-1]

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        # Use [CLS] token embedding as representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(batch_size, num_images, self.latent_dim)
        return cls_embeddings
