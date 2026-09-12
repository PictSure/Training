import math
import os
from collections import defaultdict

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import trange

from dataset.hierarchical_loader import (
    HierarchicalDuckDBEpisodicDatasetCashed,
    collate_hierarchical_episodes,
)
from model.model_PictSure import CustomTransformerModel
from utils.data_loader_imagenet import normalize_samples
from utils.dataset_factory import DatasetFactory
from utils.lr_scheduler import CustomLRScheduler
from utils.model_factory import ModelFactory
from utils.summary_writer import SummaryWriter, find_latest_run_directory


def _infer_embedding_dim(dataset):
    peek_iter = iter(dataset)
    try:
        sample = next(peek_iter)
        context_images = sample[0]
        return context_images.shape[-1]
    except StopIteration as exc:
        raise RuntimeError(
            "Unable to infer embedding dimension from DuckDB dataset; no samples available"
        ) from exc
    finally:
        close = getattr(peek_iter, "close", None)
        if callable(close):
            close()


def _estimate_num_batches(loader):
    dataset_len = None
    try:
        dataset_len = len(loader.dataset)
    except TypeError:
        dataset_len = None

    if dataset_len is not None and loader.batch_size is not None:
        return math.ceil(dataset_len / loader.batch_size)

    try:
        return len(loader)
    except TypeError:
        return None


class Trainer:
    def __init__(self, config, device, args):
        self.config = config
        self.device = device
        self.args = args
        self.resample_rate = config.get("resample", 30)
        self.best_loss = float("inf")
        self.best_acc = 0
        self.start_epoch = 0
        self.EPOCHS = self.config["optimizer"]["epochs"]
        self.using_duckdb = bool(self.config.get("duckdb-path"))
        self.encoder_name = None
        if self.using_duckdb:
            self._setup_duckdb_components()
        else:
            self.training_loader, self.test_loader = DatasetFactory(
                config, args, self.start_epoch
            ).get_dataloaders()
            try:
                self.config["embedding_dim"] = self.training_loader.dataset.embedding_dim
            except Exception:
                self.config["embedding_dim"] = None
            self.model, self.encoder_name = ModelFactory(config, device).create_model()
            self._setup_optimizer()
        self._setup_writer_and_checkpoint()
        self._setup_lrschedule()
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["optimizer"]["epsilon"])
        self.losses = []
        self.accuracies = []
        self.test_accuracies = []
        self.writer.log_hyperparameters(config)


    def _setup_optimizer(self):
        config = self.config
        if config.get("encoder"):
            encoder_params = list(self.model.embedding.parameters())
            encoder_param_ids = {id(param) for param in encoder_params}
            other_params = [param for param in self.model.parameters() if id(param) not in encoder_param_ids]
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            self.optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': float(config["optimizer"]["lr_encoder"])},
                {'params': other_params, 'lr': float(config["optimizer"]["lr_rest"])}
            ], weight_decay=float(config["optimizer"]["weight_decay"]))
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(config["optimizer"]["lr_rest"]),
                weight_decay=float(config["optimizer"]["weight_decay"]),
            )

    def _setup_duckdb_components(self):
        config = self.config
        dataset_weights = config.get("dataset_weights", None)
        if dataset_weights is not None:
            if not isinstance(dataset_weights, dict):
                raise ValueError(
                    "dataset_weights must be a dictionary mapping dataset names to weights"
                )

            total_weight = sum(dataset_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    "dataset_weights must sum to 1.0, "
                    f"but got sum={total_weight:.6f}. Provided weights: {dataset_weights}"
                )

        dataset = HierarchicalDuckDBEpisodicDatasetCashed(
            db_path=config["duckdb-path"],
            num_classes=config["dataloader"]["num_classes"],
            samples_per_class=config["dataloader"]["num_images"],
            episodes=config["dataloader"]["num_samples"],
            dataset_weights=dataset_weights,
            device="cpu",
            return_dataset_name=True,
        )

        embedding_dim = _infer_embedding_dim(dataset)
        self.config["embedding_dim"] = embedding_dim

        self.training_loader = DataLoader(
            dataset,
            batch_size=config["dataloader"]["batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            collate_fn=collate_hierarchical_episodes,
            pin_memory=False,
            prefetch_factor=1,
        )
        self.test_loader = None
        self.model = CustomTransformerModel(
            embedding_dim=embedding_dim,
            num_classes=config["dataloader"]["num_classes"],
            nheads=config["model"]["nheads"],
            nlayer=config["model"]["nlayers"],
            embed_dim=config["model"]["embed_dim"],
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["optimizer"]["lr_rest"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )

    def _setup_lrschedule(self):
        if not self.config["optimizer"]["lr_schedule"]:
            self.scheduler = None
            return

        default_warmup = self.EPOCHS * 0.2
        default_plateau = self.EPOCHS * 0.05

        warmup_epochs = max(float(self.config["optimizer"].get("warmup_epochs", default_warmup)), 0.0)
        plateau_epochs = max(float(self.config["optimizer"].get("plateau_epochs", default_plateau)), 0.0)

        base_lr = float(self.config["optimizer"].get("lr", 5e-4))
        lr_max = float(self.config["optimizer"].get("lr_initial", base_lr))
        lr_min = float(
            self.config["optimizer"].get("lr_target", lr_max * 0.1 if lr_max > 0 else base_lr * 0.1)
        )

        scheduler_kwargs = {
            "optimizer": self.optimizer,
            "epochs": self.EPOCHS,
            "warmup_epochs": warmup_epochs,
            "plateau_epochs": plateau_epochs,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "last_epoch": self.start_epoch - 1,
        }

        if self.using_duckdb:
            self.scheduler = CustomLRScheduler(**scheduler_kwargs)
        else:
            if self.config.get("encoder") is not None:
                self.scheduler = CustomLRScheduler(param_group_index=1, **scheduler_kwargs)
            else:
                self.scheduler = CustomLRScheduler(**scheduler_kwargs)

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
        if self.using_duckdb:
            self._train_duckdb()
        else:
            self._train_embedding()

    def _train_duckdb(self):
        print("Starting DuckDB training")
        epoch_progress = trange(self.start_epoch, self.EPOCHS)
        config = self.config
        self.model = self.model.to(self.device)

        for epoch in range(self.start_epoch, self.EPOCHS):
            per_dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            total_correct = 0
            total_samples = 0
            total_loss = 0

            self.model.train(True)
            num_batches = _estimate_num_batches(self.training_loader)
            if num_batches is None:
                raise TypeError("Unable to determine number of batches for the training loader")

            progressbar = trange(len(self.training_loader), leave=False)
            last_batch_idx = -1
            for batch_idx, batch in enumerate(self.training_loader):
                if len(batch) == 5:
                    images, labels, pred_image, pred_label, batch_dataset_names = batch
                else:
                    images, labels, pred_image, pred_label = batch
                    batch_dataset_names = None
                last_batch_idx = batch_idx

                images, labels, pred_image, pred_label = (
                    images.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                    pred_image.to(self.device, non_blocking=True),
                    pred_label.to(self.device, non_blocking=True),
                )

                outputs = self.model.forward(images, labels, pred_image, embedd=False)
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
                    correct_mask = predicted == pred_label
                    correct = correct_mask.sum().item()
                    total = pred_label.size(0)
                    total_correct += correct
                    total_samples += total

                    if batch_dataset_names is not None:
                        for ds_name, is_correct in zip(
                            batch_dataset_names, correct_mask.tolist()
                        ):
                            stats = per_dataset_stats[ds_name]
                            stats["correct"] += int(is_correct)
                            stats["total"] += 1

                    acc = correct / total

                if batch_idx % 10 == 0:
                    self.writer.log_batch_metrics(
                        "train", batch_idx, {"loss": loss.item(), "acc": acc}
                    )
                progressbar.update()
                progressbar.set_description(
                    "Batch [{:>5d}/{:>5d}], Loss {:.4f}, Acc: {:.2f}".format(
                        batch_idx + 1, num_batches, loss.item(), acc
                    )
                )
            progressbar.close()

            batches_processed = last_batch_idx + 1
            if batches_processed == 0:
                raise RuntimeError("Training loader produced no batches")

            avg_loss = total_loss / batches_processed
            accuracy = total_correct / total_samples
            self.losses.append(avg_loss)
            self.accuracies.append(accuracy)

            dataset_accs = {
                name: stats["correct"] / stats["total"]
                for name, stats in per_dataset_stats.items()
                if stats["total"] > 0
            }

            if dataset_accs:
                readable_accs = {name: round(val, 4) for name, val in dataset_accs.items()}
                print(f"Per-dataset train accuracy: {readable_accs}")

            if batches_processed % config["optimizer"]["acc_steps"] != 0:
                clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_grad_norm = 0.0
            grad_param_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    grad_param_count += 1
            avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0

            epoch_progress.update()
            epoch_progress.set_description(
                "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f}, Avg Gradient Norm: {:.6f}".format(
                    epoch + 1, self.EPOCHS, avg_loss, accuracy, avg_grad_norm
                )
            )

            epoch_metrics = {
                "loss": avg_loss,
                "acc": accuracy,
                "avg_grad_norm": avg_grad_norm,
                "lrs": [
                    param_group["lr"]
                    if any(p.requires_grad for p in param_group["params"])
                    else None
                    for param_group in self.optimizer.param_groups
                ],
            }

            for name, acc_val in dataset_accs.items():
                epoch_metrics[f"acc_{name}"] = acc_val

            self.writer.log_epoch_metrics("train", epoch, epoch_metrics)
            self.writer.flush()

            if self.scheduler:
                self.scheduler.step()

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.writer.save_model(model=self.model, filename="best_loss_model.pt")

            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_acc": self.best_acc,
            }
            self.writer.save_checkpoint(checkpoint)

        epoch_progress.close()
        self.writer.save_model(model=self.model, filename="final_model.pt")
        self.writer.close()

    def _train_embedding(self):
        print("Starting training")
        epoch_progress = trange(self.start_epoch, self.EPOCHS)
        config = self.config
        self.model = self.model.to(self.device)
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
                images, labels, pred_image, pred_label = images.to(
                    self.device, non_blocking=True
                ), labels.to(self.device, non_blocking=True), pred_image.to(
                    self.device, non_blocking=True
                ), pred_label.to(self.device, non_blocking=True)
                if config.get("encoder") is not None:
                    images, pred_image = normalize_samples(
                        images, pred_image, resize=(224, 224), model=self.encoder_name
                    )

                    outputs = self.model.forward(images, labels, pred_image)
                else:
                    outputs = self.model.forward(images, labels, pred_image, embedd=False)

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
                    self.writer.log_batch_metrics(
                        "train", batch_idx, {"loss": loss.item(), "acc": acc}
                    )
                progressbar.update()
            progressbar.close()
            if (batch_idx + 1) % config["optimizer"]["acc_steps"] != 0:
                clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
            test_acc = None
            if self.test_loader is not None:
                test_correct = 0
                test_samples = 0
                with torch.no_grad():
                    progressbar = trange(len(self.test_loader), leave=False)
                    for images, labels, pred_image, pred_label in self.test_loader:
                        images, labels, pred_image, pred_label = images.to(
                            self.device, non_blocking=True
                        ), labels.to(self.device, non_blocking=True), pred_image.to(
                            self.device, non_blocking=True
                        ), pred_label.to(self.device, non_blocking=True)
                        if config.get("encoder") is not None:
                            images, pred_image = normalize_samples(
                                images, pred_image, resize=(224, 224), model=self.encoder_name
                            )

                            outputs = self.model.forward(images, labels, pred_image)
                        else:
                            outputs = self.model.forward(images, labels, pred_image, embedd=False)

                        predicted = torch.argmax(outputs, dim=1)
                        correct = (predicted == pred_label.view(-1)).sum().item()
                        total = pred_label.size(0)
                        test_correct += correct
                        test_samples += total
                        progressbar.update()
                    progressbar.close()
                test_acc = test_correct / test_samples if test_samples > 0 else 0
            avg_loss = total_loss / size
            accuracy = total_correct / total_samples
            if test_acc is not None:
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
            desc = "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f}".format(
                epoch + 1, self.EPOCHS, avg_loss, accuracy
            )
            if test_acc is not None:
                desc += ", Test Acc: {:.2f}".format(test_acc)
            desc += ", Avg Gradient Norm: {:.6f}".format(avg_grad_norm)
            epoch_progress.set_description(desc)
            if self.scheduler:
                self.scheduler.step()
            epoch_metrics = {
                "loss": avg_loss,
                "acc": accuracy,
                "avg_grad_norm": avg_grad_norm,
                "lrs": [
                    param_group["lr"]
                    if any(p.requires_grad for p in param_group["params"])
                    else None
                    for param_group in self.optimizer.param_groups
                ],
            }
            if test_acc is not None:
                epoch_metrics["test_acc"] = test_acc
            self.writer.log_epoch_metrics("train", epoch, epoch_metrics)
            self.writer.flush()
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.writer.save_model(model=self.model, filename="best_loss_model.pt")
            if test_acc is not None and test_acc > self.best_acc:
                self.best_acc = test_acc
                self.writer.save_model(model=self.model, filename="best_acc_model.pt")
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_acc": self.best_acc,
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
