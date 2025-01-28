import torch

def count_parameters(model):
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_parameters = sum(p.numel() for p in model.parameters())
    return all_parameters, trainable_parameters