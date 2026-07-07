# Paths
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/valid"
TEST_DIR = "../data/test"
MODEL_SAVE_PATH = "../models/best_model.pth"

# Dataset
NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = "cuda"

# Model
PRETRAINED = True
FREEZE_BACKBONE = True
MODEL_NAME = "resnet50"

# Reproducibility
SEED = 42