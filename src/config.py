import torch

# Project settings 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Alzheimer_Dataset/data/train"
TEST_DIR = "Alzheimer_Dataset/data/test"
MODEL_SAVE_PATH = "Alzheimer_Dataset/saved_models/"
RUN_NAME = "ResNet50_New_A_FineTune_Deep"

# Model Hyperparameters
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
RANDOM_SEED = 42