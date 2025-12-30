import math
import torch
from utils.data_loader_imagenet import normalize_samples, get_cluster_random_loader, get_imagenet_random_loader
from utils.util import count_parameters
from model.model_PictSure import CustomTransformerModel
from model.wrapper import ResNetWrapper, DINOV2Wrapper, CLIPWrapper, VitNetWrapper, DINOV3Wrapper
from utils.summary_writer import SummaryWriter, find_latest_run_directory
from utils.lr_scheduler import CustomLRScheduler
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import yaml
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from torchvision import models
import time
from utils.data_loader_cifar10 import get_cifar10_random_loader
from dataset.hierarchical_loader import HierarchicalDuckDBEpisodicDatasetCashed, collate_hierarchical_episodes
from torch.utils.data import DataLoader


def _infer_embedding_dim(dataset):
    peek_iter = iter(dataset)
    try:
        sample = next(peek_iter)
        context_images = sample[0]
        return context_images.shape[-1]
    except StopIteration as exc:
        raise RuntimeError("Unable to infer embedding dimension from DuckDB dataset; no samples available") from exc
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


def setup_duckdb_training(config, device):
    dataset = HierarchicalDuckDBEpisodicDatasetCashed(
        db_path=config["duckdb-path"],
        num_classes=config["dataloader"]["num_classes"],
        samples_per_class=config["dataloader"]["num_images"],
        episodes=config["dataloader"]["num_samples"],
        device="cpu",
        return_dataset_name=True,
    )

    embedding_dim = _infer_embedding_dim(dataset)

    training_loader = DataLoader(
        dataset,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_hierarchical_episodes,
        pin_memory=False,
        prefetch_factor=1,
    )
    model = CustomTransformerModel(
        embedding_dim=embedding_dim,
        num_classes=config["dataloader"]["num_classes"],
        nheads=config["model"]["nheads"],
        nlayer=config["model"]["nlayers"],
        embed_dim=config["model"]["embed_dim"],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr_rest"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    return model, optimizer, training_loader, None, None, None


def setup_embedding_training(config, device):
    if config.get("resnet"):
        pretrained = config.get("pretrained", False)
        classifier = (
            models.resnet18(pretrained=pretrained)
            if config["resnet"] == 18
            else models.resnet34(pretrained=pretrained)
            if config["resnet"] == 34
            else models.resnet50(pretrained=pretrained)
        )
        encoder = ResNetWrapper(classifier)
        encoder_name = "resnet"
    elif config.get("dinov2"):
        encoder = DINOV2Wrapper(device=device).to(device)
        encoder_name = "dinov2"
    elif config.get("dinov3"):
        encoder = DINOV3Wrapper(device=device).to(device)
        encoder_name = "dinov3"
    elif config.get("clip"):
        encoder = CLIPWrapper(device=device).to(device)
        encoder_name = "clip"
    else:
        vit_path = config["paths"].get("visnet_weights") if config.get("pretrained", False) else None
        encoder = VitNetWrapper(path=vit_path, device=device).to(device)
        encoder_name = "vit"

    model = CustomTransformerModel(
        embedding_layer=encoder,
        num_classes=config["dataloader"]["num_classes"],
        nheads=config["model"]["nheads"],
        nlayer=config["model"]["nlayers"],
        embed_dim=config["model"]["embed_dim"],
        device=device,
    )
    encoder_params = list(encoder.parameters())
    encoder_param_ids = {id(param) for param in encoder_params}
    other_params = [param for param in model.parameters() if id(param) not in encoder_param_ids]

    lr_encoder = float(config["optimizer"]["lr_encoder"])
    lr_rest = float(config["optimizer"]["lr_rest"])

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": other_params, "lr": lr_rest},
        ],
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    for param in encoder.parameters():
        param.requires_grad = False

    return model, optimizer, None, None, encoder, encoder_name

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config file', default='./configs/slurm.yaml')
    parser.add_argument('--new', '-n', help="Start training from scratch", action="store_true")
    args = parser.parse_args()


    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    if config.get("duckdb-path"):
        model, optimizer, training_loader, test_loader, encoder, encoder_name = setup_duckdb_training(config, device)
    else:
        model, optimizer, training_loader, test_loader, encoder, encoder_name = setup_embedding_training(config, device)
    print("Model created")
    model.to(device)
    # Print the number of parameters, but with . notation for better readability
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["optimizer"]["epsilon"])
    start_epoch = 0

    total_params, trainable_params = count_parameters(model)
    print(
        f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}, Share of trainable: {trainable_params / total_params:.2%}")

    best_loss = float("inf")
    best_acc = 0
    if args.new:
        writer = SummaryWriter(
            directory=config["paths"]["output"], runname=config["name"])
    else:
        # find latest/most recent run directory and load model and optimizer state
        run_dir = find_latest_run_directory(
            config["paths"]["output"], config["name"])
        checkpoint_path = os.path.join(
            config["paths"]["output"], run_dir, "checkpoint.pt")
        if run_dir and os.path.exists(checkpoint_path):
            print(f"Resuming from: {run_dir}")
            # Resume logging in the same directory
            writer = SummaryWriter(directory=config["paths"]["output"], runfolder=run_dir)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint.keys() else 10
            best_acc = checkpoint["best_acc"] if "best_acc" in checkpoint.keys() else 0.5
        else:
            print("No checkpoint found. Starting a new run...")
            writer = SummaryWriter(
                directory=config["paths"]["output"], runname=config["name"])

    resample_rate = config.get("resample", 30)

    test_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                    452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
    
    if not config.get("duckdb-path"):
        print("Creating dataloader")
        if config["training_loc"] == "cluster":
            training_loader = get_cluster_random_loader(
                root=os.path.join(config["paths"]["dataset"], config["paths"]["train"]), class_index_path=config["paths"]["class_index"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=config["dataloader"]["num_samples"], num_images=config["dataloader"]["num_images"], mini=False, num_workers=config["dataloader"]["num_workers"], ratio=config["dataloader"]["train_ratio"])
            test_loader = get_cluster_random_loader(
                root=os.path.join(config["paths"]["dataset"], config["paths"]["test"]), batch_size=config["dataloader"]["batch_size"], num_classes=5, num_samples=500, num_images=5, mini=True, num_workers=config["dataloader"]["num_workers"], ratio=config["dataloader"]["test_ratio"])
            test_loader.dataset.build_image_index()
            if not args.new and start_epoch > 0 and start_epoch % resample_rate != 0:
                training_loader.dataset.build_image_index()
        elif config["training_loc"] == "cifar":
            training_loader = get_cifar10_random_loader(
                root=os.path.join(config["paths"]["dataset"], config["paths"]["train"]), batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=config["dataloader"]["num_samples"], num_images=config["dataloader"]["num_images"], num_workers=config["dataloader"]["num_workers"], resize_to_224=True)
            test_loader = get_cifar10_random_loader(
                root=os.path.join(config["paths"]["dataset"], config["paths"]["test"]), batch_size=config["dataloader"]["batch_size"], num_classes=5, num_samples=500, num_images=5, num_workers=config["dataloader"]["num_workers"], resize_to_224=True)
        else:
            training_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=10000, num_images=config["dataloader"]["num_images"], train=True, exclude_images=test_classes, mini=False, num_workers=config["dataloader"]["num_workers"])
            test_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=10000, num_images=config["dataloader"]["num_images"], train=True, include_images=test_classes, mini=True, num_workers=config["dataloader"]["num_workers"])

        print("DataLoader created")

    if encoder is not None and start_epoch >= 100 and config["optimizer"].get("train_embed", False):
        for param in encoder.parameters():
            param.requires_grad = True

    EPOCHS = config["optimizer"]["epochs"]
    lr_schedule_enabled = config["optimizer"].get("lr_schedule", False)
    using_duckdb = bool(config.get("duckdb-path"))

    default_warmup = EPOCHS * 0.1
    default_plateau = EPOCHS * 0.05

    warmup_epochs = max(float(config["optimizer"].get("warmup_epochs", default_warmup)), 0.0)
    plateau_epochs = max(float(config["optimizer"].get("plateau_epochs", default_plateau)), 0.0)

    base_lr = float(config["optimizer"].get("lr", 5e-4))
    lr_max = float(config["optimizer"].get("lr_initial", base_lr))
    lr_min = float(config["optimizer"].get("lr_target", lr_max * 0.1 if lr_max > 0 else base_lr * 0.1))

    scheduler_kwargs = {
        "optimizer": optimizer,
        "epochs": EPOCHS,
        "warmup_epochs": warmup_epochs,
        "plateau_epochs": plateau_epochs,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "last_epoch": start_epoch - 1,
    }

    if lr_schedule_enabled and not using_duckdb:
        scheduler = CustomLRScheduler(param_group_index=1, **scheduler_kwargs)
    else:
        scheduler = CustomLRScheduler(**scheduler_kwargs)

    losses = []
    accuracies = []
    test_accuracies = []
    writer.log_hyperparameters(config)
    print("Starting training")
    epoch_progress = trange(start_epoch, EPOCHS)

    print("="*80 + "\n" + f"Warmup epochs: {warmup_epochs}\n" + f"Plateau epochs: {plateau_epochs}\n" + f"Base LR: {base_lr}\n" + f"LR Max: {lr_max}\n" + f"LR Min: {lr_min}\n" + f"Weight decay: {config['optimizer']['weight_decay']}\n" + f"Label smoothing: {config['optimizer']['epsilon']}\n" + f"Accumulation steps: {config['optimizer']['acc_steps']}\n" + f"Total epochs: {EPOCHS}\n" + "\nMODEL HYPERPARAMETERS\n" + f"Number of heads: {config['model']['nheads']}\n" + f"Number of layers: {config['model']['nlayers']}\n" + f"Embedding dimension: {config['model']['embed_dim']}\n" + f"Encoder: {encoder_name if encoder else 'None'}\n" + f"Total parameters: {total_params:,}\n" + f"Trainable parameters: {trainable_params:,}\n" + "="*80 + "\n")

    for epoch in range(start_epoch, EPOCHS):
        per_dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        print(f"Epoch {epoch+1}/{EPOCHS} with lr_encoder={optimizer.param_groups[0]['lr']}")
        total_correct = 0
        total_samples = 0
        total_loss = 0
        if epoch % resample_rate == 0 and config["training_loc"] == "cluster":
            training_loader.dataset.build_image_index()

        model.train(True)
        num_batches = _estimate_num_batches(training_loader)
        if num_batches is None:
            raise TypeError("Unable to determine number of batches for the training loader")

        progressbar = trange(len(training_loader), leave=False)
        last_batch_idx = -1
        for batch_idx, batch in enumerate(training_loader):
            if len(batch) == 5:
                images, labels, pred_image, pred_label, batch_dataset_names = batch
            else:
                images, labels, pred_image, pred_label = batch
                batch_dataset_names = None
            last_batch_idx = batch_idx
            images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
            
            if not config.get("duckdb-path"):
                images, pred_image = normalize_samples(
                    images, pred_image, resize=(224, 224), model=encoder_name)

                outputs = model.forward(images, labels, pred_image)
            else:
                outputs = model.forward(images, labels, pred_image, embedd=False)

            pred_label = pred_label.view(-1)
            loss = loss_fn(outputs, pred_label)

            loss = loss / config["optimizer"]["acc_steps"]
            loss.backward()         # Backpropagation

            if (batch_idx + 1) % config["optimizer"]["acc_steps"] == 0:

                clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()        # Update parameters
                optimizer.zero_grad()

            total_loss += loss.item()

            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct_mask = (predicted == pred_label)
                correct = correct_mask.sum().item()
                total = pred_label.size(0)
                total_correct += correct
                total_samples += total

                if batch_dataset_names is not None:
                    for ds_name, is_correct in zip(batch_dataset_names, correct_mask.tolist()):
                        stats = per_dataset_stats[ds_name]
                        stats["correct"] += int(is_correct)
                        stats["total"] += 1

                acc = correct / total
            if batch_idx % 10 == 0:
                writer.log_batch_metrics("train", batch_idx, {
                    "loss": loss.item(), "acc": acc
                })
            progressbar.update()
            progressbar.set_description(
                "Batch [{:>5d}/{:>5d}], Loss {:.4f}, Acc: {:.2f}".format(
                    batch_idx+1, num_batches, loss.item(), acc
                )
            )
        progressbar.close()

        batches_processed = last_batch_idx + 1
        if batches_processed == 0:
            raise RuntimeError("Training loader produced no batches")

        avg_loss = total_loss / batches_processed
        accuracy = total_correct / total_samples
        losses.append(avg_loss)
        accuracies.append(accuracy)

        dataset_accs = {
            name: stats["correct"] / stats["total"]
            for name, stats in per_dataset_stats.items()
            if stats["total"] > 0
        }

        if dataset_accs:
            readable_accs = {name: round(val, 4) for name, val in dataset_accs.items()}
            print(f"Per-dataset train accuracy: {readable_accs}")

        if batches_processed % config["optimizer"]["acc_steps"] != 0:
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        if test_loader is not None:
            test_correct = 0
            test_samples = 0
            with torch.no_grad():
                progressbar = trange(len(test_loader), leave=False)
                for images, labels, pred_image, pred_label in test_loader:
                    images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
                    images, pred_image = normalize_samples(
                        images, pred_image, resize=(224, 224))

                    outputs = model.forward(images, labels, pred_image)
                    predicted = torch.argmax(outputs, dim=1)
                    correct = (predicted == pred_label.view(-1)).sum().item()
                    total = pred_label.size(0)
                    test_correct += correct
                    test_samples += total
                    progressbar.update()
                progressbar.close()

            test_acc = test_correct / test_samples
            
            test_accuracies.append(test_acc)

            total_grad_norm = 0.0
            grad_param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    grad_param_count += 1
            avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0
            epoch_progress.update()
            epoch_progress.set_description(
                "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f}, Test Acc: {:.2f}, Avg Gradient Norm: {:.6f}".format(
                    epoch+1, EPOCHS, avg_loss, accuracy, test_acc, avg_grad_norm
                )
            )
            epoch_metrics = {
                "loss": avg_loss,
                "acc": accuracy,
                "test_acc": test_acc,
                "avg_grad_norm": avg_grad_norm,
                "lrs": [param_group["lr"] if any(p.requires_grad for p in param_group["params"]) else None for param_group in optimizer.param_groups],
            }

            for name, acc_val in dataset_accs.items():
                epoch_metrics[f"acc_{name}"] = acc_val

            writer.log_epoch_metrics("train", epoch, epoch_metrics)
            writer.flush()
        else:
            epoch_progress.update()
            epoch_progress.set_description(
                "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f}".format(
                    epoch+1, EPOCHS, avg_loss, accuracy
                )
            )
            epoch_metrics = {
                "loss": avg_loss,
                "acc": accuracy,
                "lrs": [param_group["lr"] if any(p.requires_grad for p in param_group["params"]) else None for param_group in optimizer.param_groups]
            }

            for name, acc_val in dataset_accs.items():
                epoch_metrics[f"acc_{name}"] = acc_val

            writer.log_epoch_metrics("train", epoch, epoch_metrics)
            writer.flush()
        if config["optimizer"]["lr_schedule"]:
            scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            writer.save_model(model=model, filename="best_loss_model.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_acc": best_acc
        }
        writer.save_checkpoint(checkpoint)
    epoch_progress.close()
    writer.save_model(model=model, filename="final_model.pt")
    writer.close()

