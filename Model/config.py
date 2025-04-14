import os
import torch
import numpy as np

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# Folder Paths
# =============================================================================
# Get the base directory of the project (assumes this file is at the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories relative to the project root
DATA_DIR      = os.path.join(BASE_DIR, "data")
TRAIN_DIR     = os.path.join(DATA_DIR, "Train")
TEST_DIR      = os.path.join(DATA_DIR, "Test")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
SCRIPTS_DIR   = os.path.join(BASE_DIR, "scripts")

# Create directories if they do not exist
for folder in [MODEL_DIR, RESULTS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# =============================================================================
# Device Configuration
# =============================================================================
# Use GPU if available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Hyperparameters & Configurations
# =============================================================================
config = {
    "seed": 42,                   # Random seed for reproducibility
    "data_dir": DATA_DIR,         # Root directory for data
    "train_dir": TRAIN_DIR,       # Directory for training data
    "test_dir": TEST_DIR,         # Directory for test data
    "model_dir": MODEL_DIR,       # Directory to store/load model weights
    "results_dir": RESULTS_DIR,   # Directory for saving evaluation results, plots, etc.
    "device": DEVICE,             # Computation device
    "learning_rate": 1e-3,        # Learning rate for training
    "batch_size": 16,             # Batch size for training
    "num_epochs": 100,            # Number of training epochs
    "num_workers": 4,             # Number of workers for DataLoader
    "input_channels": 7,          # Number of input channels (e.g., using multiple MRI parameters)
    "num_classes": 4              # Number of segmentation classes (e.g., background, tumor sub-types)
}

# Set the random seed for reproducibility
set_seed(config["seed"])

# =============================================================================
# Print Configuration (For Debugging)
# =============================================================================
if __name__ == "__main__":
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
