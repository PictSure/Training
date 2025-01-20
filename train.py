import torch
from utils.data_loader import get_mnist_random_loader
from model.model import CustomTransformerModel, Autoencoder, EncocderWrapper
from utils.summary_writer import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import yaml
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd

with open("./configs/local.yaml", "r") as f:
    config = yaml.load(f)


writer = SummaryWriter(directory=config.paths.output, metrics=["loss", "acc"])

training_loader = get_mnist_random_loader(
    batch_size=config.dataloader.batch_size, num_samples=10000)
validation_loader = None

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


EPOCHS = config.model.epochs
# load autoencoder
autoencoder_path = "model/autoencoder_mnist.pth"
# load autoencoder
autoencoder = Autoencoder(latent_dim=64).to(device)
autoencoder.load_state_dict(torch.load(autoencoder_path))
autoencoder.to(device)
autoencoder.eval()
encoder = autoencoder.encoder
encoder = EncocderWrapper(encoder)
encoder.to(device)

model = CustomTransformerModel(encoder, 2, device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.lr, weight_decay=config.model.weight_decay)

losses = []
accuracies = []
epoch_progress = trange(EPOCHS)
writer.log_hyperparameters(config)
print("Starting training")

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))
    total_correct = 0
    total_samples = 0
    total_loss = 0
    model.train(True)
    size = len(training_loader.dataset)
    progressbar = trange(len(training_loader), leave=False)
    for batch_idx, (images, labels, pred_image, pred_label) in enumerate(training_loader):
        images, labels, pred_image, pred_label = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True), pred_image.to(device, non_blocking=True), pred_label.to(device, non_blocking=True)
        outputs = model.forward(images, labels, pred_image)

        pred_label = pred_label.view(-1)
        loss = loss_fn(outputs, pred_label)

        optimizer.zero_grad()   # Reset gradient
        loss.backward()         # Backpropagation

        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()        # Update parameters

        total_loss += loss.item()

        with torch.no_grad():
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == pred_label).sum().item()
            total = pred_label.size(0)
            total_correct += correct
            total_samples += total
            acc = correct / total
        writer.log_batch_metric("loss", loss)
        writer.log_batch_metric("acc", acc)
        progressbar.set_description(
            '[Train] Loss: {:.4f}, Acc: {:.2f} [{:>5d}/{:>5d}]'.format(
                loss, acc, (batch_idx + 1) * len(input),
                size
            )
        )
        progressbar.update()
    progressbar.close()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    writer.log_epoch_metric("loss", avg_loss)
    writer.log_epoch_metric("acc", acc)
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
        "Epoch [{:>5d}/{:>5d}], Loss {:.4f}, Accuracy: {:.2f},  Avg Gradient Norm: {avg_grad_norm:.6f}".format(
            epoch+1, EPOCHS, avg_loss, accuracy, avg_grad_norm
        )
    )

epoch_progress.close()
writer.save_to_file()
writer.save_model(model)

smoothed_losses = pd.Series(losses).rolling(window=4).mean()
smoothed_accuracies = pd.Series(accuracies).rolling(window=4).mean()

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(smoothed_losses, label='Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.plot(smoothed_accuracies, label='Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
fig.suptitle('Loss and Accuracy per Epoch')
ax1.grid(True)

writer.save_figure(fig, "train_loss_acc.jpg")

    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = 'model_{}_{}'.format(timestamp, epoch)
    #     torch.save(model.state_dict(), model_path)

