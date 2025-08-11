from torchvision import models
from model.model_PictSure import CustomTransformerModel
from model.wrapper import VitNetWrapper, ResNetWrapper, DINOV2Wrapper, CLIPWrapper


class ModelFactory:
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def create_model(self):
        encoder, encoder_name = self._create_encoder()
        return self._create_transformer(encoder), encoder_name

    def _create_encoder(self):
        encoder = self.config.get("encoder")
        if "resnet" in encoder:
            encoder_type = encoder.split("-")[-1]
            encoder = "resnet"
            return self._create_resnet_encoder(encoder_type), encoder
        elif encoder == "dinov2":
            return DINOV2Wrapper(device=self.device).to(self.device), encoder
        elif encoder == "clip":
            return CLIPWrapper(device=self.device).to(self.device), encoder
        return self._create_vit_encoder(), encoder
    
    def _create_resnet_encoder(self, encoder_type):
        classifier = (
            models.resnet18(weights="DEFAULT")
            if encoder_type == 18
            else models.resnet34(weights="DEFAULT")
            if encoder_type == 34
            else models.resnet50(weights="DEFAULT")
        )
        encoder = ResNetWrapper(classifier)
        return encoder
    
    def _create_vit_encoder(self):
        vit_path = self.config["paths"].get("visnet_weights")
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