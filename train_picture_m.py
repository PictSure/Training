# from model.model_cifar import CustomResnetEmbedding, EmbeddingWrapper, CustomTransformerModel, ResNetWrapper
from model.model_PictSure_M import CustomTransformerModel, ResNetWrapper
from utils.data_loader_imagenet import get_imagenet_random_loader, normalize_samples
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import torchvision.models as models
from utils.util import count_parameters

device = "cuda:1"
num_classes = 5
epsilon = 0.1

# classifier = models.resnet18(pretrained=True)
# classifier = models.resnet18(pretrained=True)
classifier = models.resnet34(pretrained=True)
# classifier = CustomResnetEmbedding()

encoder = ResNetWrapper(classifier)
# encoder = EmbeddingWrapper(classifier)

criterion = nn.CrossEntropyLoss(label_smoothing=epsilon)

batch_size = 2

print("Torch precision: ", torch.get_default_dtype())

def train(lr=1e-3, num_epochs=40, num_images=10, batch_size=batch_size):
    initial_lr = 1e-6  # Start with a smaller learning rate
    target_lr = 1e-7
    model = CustomTransformerModel(encoder, num_classes, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    

    losses = []
    accuracies = []
    test_accuracies = []

    total_params, trainable_params = count_parameters(model)
    # Print the number of parameters, but with . notation for better readability
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}, Share of trainable: {trainable_params / total_params:.2%}")

    test_classes = [87, 155, 178, 181, 199, 217, 284, 321, 452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]

    print("Loading training data")
    train_loader = get_imagenet_random_loader(batch_size=batch_size, num_classes=num_classes, num_samples=10000, num_images=num_images, train=True, exclude_images=test_classes, mini=False)
    test_loader = get_imagenet_random_loader(batch_size=batch_size, num_classes=num_classes, num_samples=1000, num_images=num_images, train=False, include_images=test_classes, mini=True)

    print("Starting training")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Linear ramp up of learning rate only over the first 10 epochs
        if epoch < 60:
            current_lr = initial_lr + (lr - initial_lr) * (epoch / 60)
        elif epoch > 200:
            current_lr = lr - (lr - target_lr) * ((epoch - 200) / 500)
        else:
            current_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        total_correct = 0
        total_samples = 0
        total_loss = 0

        accumulation_steps = 8  # Number of steps to accumulate gradients

        for batch_idx, (images, labels, pred_image, pred_label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
            images, pred_image = normalize_samples(images, pred_image, sharpness=True, gaussian=True, resize=(224, 224))
            outputs = model.forward(images, labels, pred_image)
            
            pred_label = pred_label.view(-1)
            loss = criterion(outputs, pred_label)
            
            loss = loss / accumulation_steps  # Normalize loss to account for gradient accumulation
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # Multiply back the loss

            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == pred_label).sum().item()
                total = pred_label.size(0)
                total_correct += correct
                total_samples += total

        # Ensure the gradients are updated for the last few batches
        if (batch_idx + 1) % accumulation_steps != 0:
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        total_grad_norm = 0.0
        grad_param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                grad_param_count += 1

        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for images, labels, pred_image, pred_label in tqdm(test_loader, desc="Testing"):
                images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
                images, pred_image = normalize_samples(images, pred_image, resize=(224, 224))

                outputs = model.forward(images, labels, pred_image)
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == pred_label.view(-1)).sum().item()
                total = pred_label.size(0)
                test_correct += correct
                test_samples += total

        test_accuracy = test_correct / test_samples
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        test_accuracies.append(test_accuracy)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        avg_grad_norm = total_grad_norm / grad_param_count if grad_param_count > 0 else 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}, Avg Gradient Norm: {avg_grad_norm:.6f}, Learning rate: {current_lr:.6f}")

        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
        print(f"Estimated time left: {remaining_time // 60:.0f} minutes {remaining_time % 60:.0f} seconds")

    # Save the model
    model.eval()
    torch.save(model.state_dict(), f"model/model_PictSure_M.pth")
    return avg_loss, accuracy, losses, accuracies, test_accuracies


learning_rates = [1e-4]
num_epochs = 700
batches_per_epoch = 50000

results = {}

plt.figure(figsize=(10, 6))
for batch_size, lr in [(16, 1e-4)]:
    avg_loss, accuracy, losses, accuracies, test_accuracies = train(lr=lr, num_epochs=num_epochs, batch_size=batch_size, num_images=5)

    smoothed_losses = pd.Series(losses).rolling(window=2).mean()
    smoothed_accuracies = pd.Series(accuracies).rolling(window=2).mean()
    smoothed_test_accuracies = pd.Series(test_accuracies).rolling(window=2).mean()

    results[batch_size] = (smoothed_accuracies, smoothed_test_accuracies, lr)

    plt.plot(smoothed_accuracies, label=f"Train Accuracy")
    plt.plot(smoothed_test_accuracies, label=f"Test Accuracy")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch for different batch sizes')
plt.legend()
plt.grid(True)
plt.savefig("5_shot_PictSure_M.pdf")

# Save the losses and accuracies in a CSV file
df = pd.DataFrame(results)
df.to_csv("5_shot_PictSure_M.csv")