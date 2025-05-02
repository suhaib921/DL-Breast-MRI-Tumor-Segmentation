import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_name, seg_parts, device):
    """Get model based on name."""
    if model_name == "UNet":
        from unet.unet_model import UNet
        return UNet(in_channels=3, out_channels=seg_parts).to(device)
    else:
        raise ValueError(f"Model {model_name} not implemented")

class JointTransform:
    """Applies the same spatial augmentations to both image and mask."""
    def __init__(self, augment=True, size=(300, 300)):
        self.augment = augment
        self.size = size

    def __call__(self, image, mask):
        # Resize to specified size
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
        mask = (mask > 0.5).float() # Normalize image and binarize mask


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

# Modified BreastDataset_seg class in utils.py
class BreastDatasetSeg(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir: Path to directory with PNG files
            split: One of 'train', 'val', 'test'
            transform: Albumentations transforms
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._get_samples(split)
        
    def _get_samples(self, split):
        """Get list of (post_img_path, subtraction_img_path, mask_path) tuples"""
        samples = []
        # Implement your split logic here (e.g., using text files)
        # Example structure:
        for case_id in sorted(os.listdir(self.data_dir)):
            case_dir = os.path.join(self.data_dir, case_id)
            post_img = os.path.join(case_dir, "post_contrast.png")
            subtraction_img = os.path.join(case_dir, "subtraction.png")
            mask_img = os.path.join(case_dir, "mask.png")
            
            if all([os.path.exists(post_img), 
                    os.path.exists(substitution_img),
                    os.path.exists(mask_img)]):
                samples.append((post_img, subtraction_img, mask_img))
        return samples[:int(0.8*len(samples))]  # Example split
    
    def __getitem__(self, idx):
        post_path, sub_path, mask_path = self.samples[idx]
        
        # Load images
        post_img = np.array(Image.open(post_path).convert("L"))  # Grayscale
        sub_img = np.array(Image.open(sub_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Stack inputs: (H, W, 2) for post-contrast + subtraction
        image = np.stack([post_img, sub_img], axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (2, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
        
        return image, mask
# Transforms for data augmentation that match the notebook
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandFlip:
    def __init__(self, spatial_axis, prob=0.5):
        self.spatial_axis = spatial_axis
        self.prob = prob
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            for axis in self.spatial_axis:
                img = torch.flip(img, [axis])
                mask = torch.flip(mask, [axis])
        return img, mask

class RandRotate:
    def __init__(self, range_x, prob=0.5):
        self.range_x = range_x
        self.prob = prob
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            angle = random.uniform(-self.range_x, self.range_x)
            # Convert to PIL for rotation
            if not isinstance(img, Image.Image):
                img_pil = transforms.ToPILImage()(img)
                mask_pil = transforms.ToPILImage()(mask)
            else:
                img_pil, mask_pil = img, mask
                
            img_rotated = F.rotate(img_pil, angle)
            mask_rotated = F.rotate(mask_pil, angle)
            
            # Convert back to tensor
            if not isinstance(img, Image.Image):
                img = transforms.ToTensor()(img_rotated)
                mask = transforms.ToTensor()(mask_rotated)
            else:
                img, mask = img_rotated, mask_rotated
                
        return img, mask