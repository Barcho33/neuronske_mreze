import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_dataloaders
from model import get_resnet50
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    torch.manual_seed(config.SEED)

    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Korišćeni uređaj za trening: {device}")

    print("Učitavanje data loader-a...")
    train_loader, val_loader, _ = get_dataloaders()

    print(f"Inicijalizacija modela {config.MODEL_NAME} za Feature Extraction...")
    model = get_resnet50(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        trainable_params,
        lr=config.LEARNING_RATE
    )

    best_val_loss = float("inf")

    os.makedirs(
        os.path.dirname(config.MODEL_SAVE_PATH),
        exist_ok=True
    )

    print("\n--- Početak treninga (Feature Extraction) ---")

    for epoch in range(config.EPOCHS):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoha [{epoch + 1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Sačuvan novi najbolji model u: {config.MODEL_SAVE_PATH}")

    print("\nTrening uspešno završen!")


if __name__ == "__main__":
    main()