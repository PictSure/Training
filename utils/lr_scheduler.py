import torch
import math

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, last_epoch=-1):
        self.epochs = epochs
        self.lr_max = 5e-4
        self.lr_min = 5e-5
        self.warmup_epochs = 50
        self.plateau_epochs = 20
        self.decay_epochs = epochs - (self.warmup_epochs + self.plateau_epochs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1  # Adjust to match human-readable epochs (1-based)
        
        if epoch <= self.warmup_epochs:
            # Linear warmup: start at 0, go to lr_max
            return [self.lr_max * (epoch / self.warmup_epochs) for _ in self.base_lrs]

        elif epoch <= self.warmup_epochs + self.plateau_epochs:
            # Plateau: stay at lr_max
            return [self.lr_max for _ in self.base_lrs]

        else:
            # Logarithmic decay from lr_max to lr_min
            decay_epoch = epoch - (self.warmup_epochs + self.plateau_epochs)
            decay_factor = (decay_epoch / self.decay_epochs)  # Normalize
            log_lr = self.lr_max * (self.lr_min / self.lr_max) ** decay_factor
            return [log_lr for _ in self.base_lrs]