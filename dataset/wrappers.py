import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
import os

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of images with shape (batch_size, 3, H, W)
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=x)
        # Use [CLS] token embedding as representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

class DINOV3Wrapper(nn.Module):
    """CLS-only wrapper for facebook/dinov3-vith16plus-pretrain-lvd1689m."""

    def __init__(self, device: str = "cpu", use_device_map: bool = False):
        super().__init__()
        self.device = device

        kwargs = {"token": os.getenv("HF_TOKEN")}
        if use_device_map:
            kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(
                "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                **kwargs,
            )
        else:
            self.model = AutoModel.from_pretrained(
                "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                **kwargs,
            ).to(device)

        self.model.eval()
        self.latent_dim = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_images, 3, H, W) already normalized for DINOv3.
        Returns CLS embeddings shaped (batch, num_images, hidden_size).
        """

        bsz, nimg, _, h, w = x.shape
        flat = x.reshape(-1, 3, h, w)

        if flat.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            flat = flat.float()

        if hasattr(self.model, "device"):
            target_device = self.model.device
        else:
            target_device = next(self.model.parameters()).device

        flat = flat.to(target_device, non_blocking=True)

        outputs = self.model(pixel_values=flat)
        last_hidden = outputs.last_hidden_state  # (bsz*nimg, 1+reg+patches, hidden)

        cls_tokens = last_hidden[:, 0, :]

        # Optional debugging hook to ensure patch alignment when shapes change.
        if (h % self.patch_size) != 0 or (w % self.patch_size) != 0:
            raise ValueError("Image size must be divisible by patch size for DINOv3")

        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        expected_tokens = 1 + self.num_register_tokens + num_patches_h * num_patches_w
        if last_hidden.shape[1] != expected_tokens:
            raise RuntimeError("Unexpected token layout in DINOv3 last_hidden_state")

        return cls_tokens.reshape(bsz, nimg, self.latent_dim)
    
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