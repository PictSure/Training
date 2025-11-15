import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.model_ViT import VisionTransformer
import os
    
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

class DINOV3Wrapper(nn.Module):
    """
    Wrapper for 'facebook/dinov3-vith16plus-pretrain-lvd1689m' that **assumes inputs are already
    preprocessed** (rescaled to [0,1], resized to 224x224 (or model-acceptable size), and
    normalized with ImageNet mean/std in channels-first format).

    Expected input to forward:
        x: Tensor of shape (batch, num_images, 3, H, W)  -- pre-normalized.
    Returns:
        CLS embeddings of shape (batch, num_images, latent_dim)
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            token=os.getenv("HF_TOKEN")
        ).to(device)
        self.model.eval()

        # Probe latent dim using a dummy pixel_values tensor (already "normalized" zeros).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=torch.float32)
            out = self.model(pixel_values=dummy)
        self.latent_dim = out.last_hidden_state.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_images, 3, H, W) -- already preprocessed/normalized.
        """
        bsz, nimg = x.size(0), x.size(1)

        # Flatten to (batch*num_images, 3, H, W)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        # Ensure device/dtype are suitable for the model
        if x.dtype != torch.float32:
            x = x.float()
        x = x.to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model(pixel_values=x)  # bypass processor; pixel_values expected
            cls = outputs.last_hidden_state[:, 0, :]  # (batch*nimg, dim)

        cls = cls.view(bsz, nimg, self.latent_dim)
        return cls

class CLIPWrapper(nn.Module):
    def __init__(self, device="cpu"):
        super(CLIPWrapper, self).__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.device = device
        # Get latent dim by running a dummy input through the model
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=dummy)
            image_embeds = vision_outputs[1]
            image_embeds = self.model.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        self.latent_dim = image_embeds.shape[-1]

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=x)
            image_embeds = vision_outputs[1]
            image_embeds = self.model.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds.view(batch_size, num_images, self.latent_dim)
        return image_embeds

class VitNetWrapper(nn.Module):
    def __init__(self, path, device, num_classes=1000):
        super().__init__()
        self.embedding = VisionTransformer(num_classes=num_classes)
        if path:
            self.embedding.load_state_dict(torch.load(path, map_location=device))
        self.latent_dim = self.embedding.embed_dim

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.embedding.forward(x)[1]
        x = x.view(batch_size, num_images, self.latent_dim)
        return x