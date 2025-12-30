import torch
import math

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self, 
            optimizer, 
            epochs,
            param_group_index=None, 
            lr_max=5e-4, 
            lr_min = 5e-5, 
            warmup_epochs=50, 
            plateau_epochs=10, 
            last_epoch=-1
    ):
        self.epochs = epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epochs
        self.decay_epochs = max(epochs - (self.warmup_epochs + self.plateau_epochs), 0.0)
        if param_group_index is None:
            self.param_group_index = "all"
        elif isinstance(param_group_index, int):
            self.param_group_index = [param_group_index]
        else:
            self.param_group_index = param_group_index
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1  # Adjust to match human-readable epochs (1-based)
        
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            # Linear warmup: start at 0, go to lr_max
            scheduled_lr = self.lr_max * (epoch / self.warmup_epochs)
        elif epoch <= self.warmup_epochs + self.plateau_epochs:
            # Plateau: stay at lr_max
            scheduled_lr = self.lr_max
        else:
            # Logarithmic decay from lr_max to lr_min
            decay_epoch = epoch - (self.warmup_epochs + self.plateau_epochs)
            if self.decay_epochs <= 0:
                scheduled_lr = self.lr_min
            else:
                decay_factor = (decay_epoch / self.decay_epochs)  # Normalize
                scheduled_lr = self.lr_max * (self.lr_min / self.lr_max) ** decay_factor

        new_lrs = []
        
        for i, group in enumerate(self.optimizer.param_groups):
            if self.param_group_index == "all" or i in self.param_group_index:
                new_lrs.append(scheduled_lr)
            else:
                new_lrs.append(group['lr'])  # keep current
        return new_lrs