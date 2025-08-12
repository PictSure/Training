import torch
from utils.data_loader_imagenet import normalize_samples
from utils.summary_writer import SummaryWriter, find_latest_run_directory
from utils.lr_scheduler import CustomLRScheduler
from torch.nn.utils import clip_grad_norm_
import yaml
from tqdm import trange
import os
from utils.model_factory import ModelFactory
from utils.dataset_factory import DatasetFactory


class Trainer:
    def __init__(self, config, device, args):
        self.config = config
        self.device = device
        self.args = args
        self.resample_rate = config.get("resample", 30)
        self.test_classes = [87, 155, 178, 181, 199, 217, 284, 321, 452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
        self.best_loss = float("inf")
        self.best_acc = 0
        self.start_epoch = 0
        self.model, self.encoder_name = ModelFactory(config, device).create_model()
        self._setup_optimizer()
        self._setup_writer_and_checkpoint()
        self._setup_lrschedule()
        self.training_loader, self.test_loader = DatasetFactory(config, args, self.start_epoch).get_dataloaders()
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["optimizer"]["epsilon"])
        self.losses = []
        self.accuracies = []
        self.test_accuracies = []
        self.writer.log_hyperparameters(config)

    def _setup_optimizer(self):
        config = self.config
        encoder_params = list(self.model.embedding.parameters())
        encoder_param_ids = {id(param) for param in encoder_params}
        other_params = [param for param in self.model.parameters() if id(param) not in encoder_param_ids]
        for param in self.model.embedding.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': float(config["optimizer"]["lr_encoder"])},
            {'params': other_params, 'lr': float(config["optimizer"]["lr_rest"])}
        ], weight_decay=float(config["optimizer"]["weight_decay"]))

    def _setup_lrschedule(self):
        self.EPOCHS = config["optimizer"]["epochs"]
        if config["optimizer"]["lr_schedule"]:
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                epochs=self.EPOCHS,
                param_group_index=1,
                last_epoch=self.start_epoch-1
            )
        else:
            self.scheduler = None

    def _setup_writer_and_checkpoint(self):
        config = self.config
        args = self.args
        if args.new:
            self.writer = SummaryWriter(directory=config["paths"]["output"], runname=config["name"])
        else:
            run_dir = find_latest_run_directory(config["paths"]["output"], config["name"])
            checkpoint_path = os.path.join(config["paths"]["output"], run_dir, "checkpoint.pt")
            if run_dir and os.path.exists(checkpoint_path):
                print(f"Resuming from: {run_dir}")
                self.writer = SummaryWriter(directory=config["paths"]["output"], runfolder=run_dir)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.start_epoch = checkpoint["epoch"]
                self.best_loss = checkpoint.get("best_loss", 10)
                self.best_acc = checkpoint.get("best_acc", 0.5)
            else:
                print("No checkpoint found. Starting a new run...")
                self.writer = SummaryWriter(directory=config["paths"]["output"], runname=config["name"])

    def train(self):
        print("Starting training")
        epoch_progress = trange(self.start_epoch, self.EPOCHS)
        config = self.config
        for epoch in range(self.start_epoch, self.EPOCHS):
            total_correct = 0
            total_samples = 0
            total_loss = 0
            if epoch % self.resample_rate == 0 and config["training_loc"] == "cluster":
                self.training_loader.dataset.build_image_index()
            self.model.train(True)
            size = len(self.training_loader)
            progressbar = trange(len(self.training_loader), leave=False)
            for batch_idx, (images, labels, pred_image, pred_label) in enumerate(self.training_loader):
                images, labels, pred_image, pred_label = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True), pred_image.to(self.device, non_blocking=True), pred_label.to(self.device, non_blocking=True)
                images, pred_image = normalize_samples(images, pred_image, resize=(224, 224), model=self.encoder_name)

                outputs = self.model.forward(images, labels, pred_image)

                pred_label = pred_label.view(-1)
                loss = self.loss_fn(outputs, pred_label)
                
                loss = loss / config["optimizer"]["acc_steps"]
                loss.backward()
                
                if (batch_idx + 1) % config["optimizer"]["acc_steps"] == 0:
                    clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                with torch.no_grad():
                    predicted = torch.argmax(outputs, dim=1)
                    correct = (predicted == pred_label).sum().item()
                    total = pred_label.size(0)
                    total_correct += correct
                    total_samples += total
                    acc = correct / total
                if batch_idx % 10 == 0:
                    self.writer.log_batch_metrics("train", batch_idx, {"loss": loss.item(), "acc": acc})
                progressbar.update()
            progressbar.close()
            if (batch_idx + 1) % config["optimizer"]["acc_steps"] != 0:
                clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
            test_correct = 0
            test_samples = 0
            with torch.no_grad():
                progressbar = trange(len(self.test_loader), leave=False)
                for images, labels, pred_image, pred_label in self.test_loader:
                    images, labels, pred_image, pred_label = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True), pred_image.to(self.device, non_blocking=True), pred_label.to(self.device, non_blocking=True)
                    images, pred_image = normalize_samples(images, pred_image, resize=(224, 224), model=self.encoder_name)

                    outputs = self.model.forward(images, labels, pred_image)

                    predicted = torch.argmax(outputs, dim=1)
                    correct = (predicted == pred_label.view(-1)).sum().item()
                    total = pred_label.size(0)
                    test_correct += correct
                    test_samples += total
                    progressbar.update()
                progressbar.close()
            test_acc = test_correct / test_samples
            avg_loss = total_loss / size
            accuracy = total_correct / total_samples
            self.test_accuracies.append(test_acc)
            self.losses.append(avg_loss)
            self.accuracies.append(accuracy)
            total_grad_norm = 0.0
            grad_param_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    grad_param_count += 1
            avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0
            epoch_progress.update()
            epoch_progress.set_description(
                "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f}, Test Acc: {:.2f}, Avg Gradient Norm: {:.6f}".format(
                    epoch+1, self.EPOCHS, avg_loss, accuracy, test_acc, avg_grad_norm
                )
            )
            if self.scheduler:
                self.scheduler.step()
            self.writer.log_epoch_metrics("train", epoch, {
                "loss": avg_loss, "acc": accuracy, "test_acc": test_acc, "avg_grad_norm": avg_grad_norm, "lrs": [param_group["lr"] if any(p.requires_grad for p in param_group["params"]) else None for param_group in self.optimizer.param_groups]
            })
            self.writer.flush()
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.writer.save_model(model=self.model, filename="best_loss_model.pt")
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.writer.save_model(model=self.model, filename="best_acc_model.pt")
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_acc": self.best_acc
            }
            self.writer.save_checkpoint(checkpoint)
        epoch_progress.close()
        self.writer.save_model(model=self.model, filename="final_model.pt")
        self.writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config file', default='./configs/slurm.yaml')
    parser.add_argument('--new', '-n', help="Start training from scratch", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    trainer = Trainer(config, device, args)
    trainer.train()
