import torch
from utils.data_loader_imagenet import normalize_samples, get_cluster_random_loader
from utils.util import count_parameters
from model.model_cifar import CustomTransformerModel, EmbeddingWrapper, ResNetWrapper
from utils.summary_writer import SummaryWriter
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
    args = parser.parse_args()


    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    writer = SummaryWriter(directory=config["paths"]["output"], runname=config["name"])

    test_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                    452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
    print("Creating dataloader")
    training_loader = get_cluster_random_loader(
        root=os.path.join(config["paths"]["dataset"], "imagenet-train2.msgpack"), batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=10000, num_images=config["dataloader"]["num_images"], mini=False, num_workers=0, ratio=0.05)
    test_loader = get_cluster_random_loader(
        root=os.path.join(config["paths"]["dataset"], "imagenet-test2.msgpack"), batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=500, num_images=config["dataloader"]["num_images"], mini=True, num_workers=0, ratio=1)
    test_loader.dataset.build_image_index()
    print("DataLoader created")
    # training_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"], num_samples=10000,
    #                                              num_images=config["dataloader"]["num_images"], train=True, exclude_images=test_classes, mini=False, num_workers=config["dataloader"]["worker"])
    # test_loader = get_imagenet_random_loader(root=config["paths"]["dataset"], batch_size=config["dataloader"]["batch_size"], num_classes=config["dataloader"]["num_classes"],
    #                                          num_samples=10000, num_images=config["dataloader"]["num_images"], train=True, include_images=test_classes, mini=True, num_workers=config["dataloader"]["worker"])
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")


    EPOCHS = config["optimizer"]["epochs"]

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
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}, Share of trainable: {trainable_params / total_params:.2%}")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["optimizer"]["epsilon"])
    lr = config["optimizer"]["lr"]
    target_lr = config["optimizer"]["lr_target"]
    initial_lr = config["optimizer"]["lr_initial"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=config["optimizer"]["weight_decay"])

    losses = []
    accuracies = []
    test_accuracies = []
    writer.log_hyperparameters(config)
    print("Starting training")
    epoch_progress = trange(EPOCHS)

    for epoch in range(EPOCHS):
        if epoch < 30:
            current_lr = initial_lr + (lr - initial_lr) * (epoch / 30)
        elif epoch > 150:
            current_lr = lr - (lr - target_lr) * ((epoch - 150) / 150)
        else:
            current_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        total_correct = 0
        total_samples = 0
        total_loss = 0
        if epoch % 5 == 0:
            training_loader.dataset.build_image_index()

        model.train(True)
        size = len(training_loader.dataset)
        progressbar = trange(len(training_loader), leave=False)
        for batch_idx, (images, labels, pred_image, pred_label) in enumerate(training_loader):
            start = time.time()
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
            writer.log_batch_metrics("train", batch_idx, {
                "loss": loss.item(), "acc": acc
            })
            end = time.time()
            progressbar.set_description(
                '[Train] Loss: {:.4f}, Acc: {:.2f} [Batch: {:>5d}, Total Time: {:.4f}s]'.format(
                    loss, acc, (batch_idx + 1), (end - start)
                )
            )
            progressbar.update()
        progressbar.close()

        if (batch_idx + 1) % config["optimizer"]["acc_steps"] != 0:
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
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
        avg_loss = total_loss / total_samples
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
    epoch_progress.close()
    writer.save_model(model=model, epoch_idx=epoch)

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

