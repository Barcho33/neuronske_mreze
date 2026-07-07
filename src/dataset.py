import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from transforms import train_transform, val_transform


def get_train_dataset():
    return datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )


def get_validation_dataset():
    return datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform
    )


def get_test_dataset():
    return datasets.ImageFolder(
        root=TEST_DIR,
        transform=val_transform
    )


def get_dataloaders():
    train_dataset = get_train_dataset()
    val_dataset = get_validation_dataset()
    test_dataset = get_test_dataset()

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def get_class_names():
    dataset = get_train_dataset()
    return dataset.classes


def get_class_to_idx():
    dataset = get_train_dataset()
    return dataset.class_to_idx


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    print("Classes:", get_class_names())
    print("Class to index:", get_class_to_idx())

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)