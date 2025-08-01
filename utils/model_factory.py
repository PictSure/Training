from torchvision import models
from model.model_PictSure import CustomTransformerModel, ResNetWrapper
from model.model_ViT import VitNetWrapper

class ModelFactory:
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def create_model(self):
        encoder = self._create_encoder()
        return self._create_transformer(encoder)

    def _create_encoder(self):
        if self.config.get("resnet"):
            return self._create_resnet_encoder()
        return self._create_vit_encoder()
    
    def _create_resnet_encoder(self):
        pretrained = self.config.get("pretrained", False)
        classifier = (
            models.resnet18(weights="DEFAULT" if pretrained else None)
            if self.config["resnet"] == 18
            else models.resnet34(weights="DEFAULT" if pretrained else None)
            if self.config["resnet"] == 34
            else models.resnet50(weights="DEFAULT" if pretrained else None)
        )
        encoder = ResNetWrapper(classifier)
        return encoder
    
    def _create_vit_encoder(self):
        vit_path = self.config["paths"].get("visnet_weights") if self.config.get("pretrained", False) else None
        encoder = VitNetWrapper(path=vit_path, device=self.device).to(self.device)
        return encoder
    
    def _create_transformer(self, encoder):
        return CustomTransformerModel(
            encoder,
            self.config["dataloader"]["num_classes"],
            nheads=self.config["model"]["nheads"],
            nlayer=self.config["model"]["nlayers"],
            device=self.device
        )