import torch
from utils.data_loader_imagenet import normalize_samples, get_cluster_random_loader
from utils.util import count_parameters
from model.model_PictSure import CustomTransformerModel, EmbeddingWrapper, ResNetWrapper
from utils.summary_writer import SummaryWriter, find_latest_run_directory
from utils.lr_scheduler import CustomLRScheduler
from torch.nn.utils import clip_grad_norm_
import yaml
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from torchvision import models
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config file', default='./configs/slurm.yaml')
    parser.add_argument('--new', '-n', help="Start training from scratch", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--vc', help="If flag is set, dataset class for variable number of classes is used", action="store_true")
    group.add_argument("-s", "--semantic", help="Use SemanticSimilarity Dataloader", action="store_true")
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
    classifier = (
        models.resnet18(pretrained=True)
        if config["resnet"] == 18
        else models.resnet34(pretrained=True)
        if config["resnet"] == 34
        else models.resnet50(pretrained=True)
    )
    encoder = ResNetWrapper(classifier)

    model = CustomTransformerModel(encoder, config["dataloader"]
                                   ["num_classes"], nheads=config["model"]["nheads"], nlayer=config["model"]["nlayers"], device=device)
    print("Model created")
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    # Print the number of parameters, but with . notation for better readability
    print(
        f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}, Share of trainable: {trainable_params / total_params:.2%}")
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["optimizer"]["epsilon"])
    lr = config["optimizer"]["lr"]
    target_lr = config["optimizer"]["lr_target"]
    initial_lr = config["optimizer"]["lr_initial"]
    optimizer = torch.optim.AdamW(model.parameters(
    ), lr=initial_lr, weight_decay=config["optimizer"]["weight_decay"])

    start_epoch = 0
    best_loss = float("inf")
    best_acc = 0
    if args.new:
        writer = SummaryWriter(
            directory=config["paths"]["output"], runname=config["name"])
    else:
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



    test_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                    452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
    print("Creating dataloader")
    training_loader = get_cluster_random_loader(
        root=os.path.join(config["paths"]["dataset"], config["paths"]["train"]), batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=config["dataloader"]["num_samples"], num_images=config["dataloader"]["num_images"], mini=False, num_workers=config["dataloader"]["num_workers"], ratio=config["dataloader"]["train_ratio"], vc=args.vc, semantic=args.semantic, hierarchy_path=config["paths"].get("hierarchy_path"), class_index_path=config["paths"].get("class_index_path"))
    test_loader = get_cluster_random_loader(
        root=os.path.join(config["paths"]["dataset"], config["paths"]["test"]), batch_size=config["dataloader"]["batch_size"], num_classes=5, num_samples=500, num_images=5, mini=True, num_workers=config["dataloader"]["num_workers"], ratio=config["dataloader"]["test_ratio"])
    test_loader.dataset.build_image_index()
    if not args.new and not args.semantic and start_epoch > 0 and start_epoch % 30 != 0:
        training_loader.dataset.build_image_index()
    print("DataLoader created")
    # training_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=10000,
    #                                              num_images=config["dataloader"]["num_images"], train=True, exclude_images=test_classes, mini=False, num_workers=config["dataloader"]["worker"])
    # test_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"],
    #                                          num_samples=10000, num_images=config["dataloader"]["num_images"], train=True, include_images=test_classes, mini=True, num_workers=config["dataloader"]["worker"])


    EPOCHS = config["optimizer"]["epochs"]
    scheduler = CustomLRScheduler(optimizer, epochs=EPOCHS, last_epoch=start_epoch-1)

    losses = []
    accuracies = []
    test_accuracies = []
    writer.log_hyperparameters(config)
    print("Starting training")
    epoch_progress = trange(start_epoch, EPOCHS)

    for epoch in range(start_epoch, EPOCHS):
        
        total_correct = 0
        total_samples = 0
        total_loss = 0
        if epoch % 30 == 0 and not args.semantic:
            training_loader.dataset.build_image_index()
        elif args.semantic:
            training_loader.dataset.resample()

        model.train(True)
        size = len(training_loader)
        progressbar = trange(len(training_loader), leave=False)
        for batch_idx, (images, labels, pred_image, pred_label) in enumerate(training_loader):
            images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
            images, pred_image = normalize_samples(
                images, pred_image, resize=(224, 224))
            
            outputs = model.forward(images, labels, pred_image)

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
                correct = (predicted == pred_label).sum().item()
                total = pred_label.size(0)
                total_correct += correct
                total_samples += total
                acc = correct / total
            if batch_idx % 10 == 0:
                writer.log_batch_metrics("train", batch_idx, {
                    "loss": loss.item(), "acc": acc
                })
            progressbar.update()
        progressbar.close()

        if (batch_idx + 1) % config["optimizer"]["acc_steps"] != 0:
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
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
        avg_loss = total_loss / size
        accuracy = total_correct / total_samples
        
        test_accuracies.append(test_acc)
        losses.append(avg_loss)
        accuracies.append(accuracy)

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
        writer.log_epoch_metrics("train", epoch, {
            "loss": avg_loss, "acc": accuracy, "test_acc": test_acc, "avg_grad_norm": avg_grad_norm
        })
        writer.flush()
        if avg_loss < best_loss:
            best_loss = avg_loss
            writer.save_model(model=model, filename="best_loss_model.pt")
        if test_acc > best_acc:
            best_acc = test_acc
            writer.save_model(model=model, filename="best_acc_model.pt")
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

    smoothed_losses = pd.Series(losses).rolling(window=4).mean()
    smoothed_accuracies = pd.Series(accuracies).rolling(window=4).mean()
    smoothed_test_accuracies = pd.Series(test_accuracies).rolling(window=2).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(smoothed_accuracies, label='Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(smoothed_test_accuracies, label='Test Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.suptitle('Accuracy vs. Epoch for different batch sizes')
    ax1.grid(True)

    writer.save_figure(fig, "train_loss_acc.jpg")
    writer.close()

