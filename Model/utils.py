import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms

class JointTransform:
    """Applies the same spatial augmentations to both image and mask."""
    def __init__(self, augment=True, size=(300, 300)):
        self.augment = augment
        self.size = size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, mask):
        # Resize to 300x300
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size)

        # Apply augmentations
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            # Random rotation (-20 to 20 degrees)
            angle = random.uniform(-20, 20)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # Normalize image and binarize mask
        image = self.normalize(image)
        mask = (mask > 0.5).float()

        return image, mask

class CustomDataset(Dataset):
    """Loads image-mask pairs from disk."""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale mask
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask