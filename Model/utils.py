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
        return UNet(in_channels=7, out_channels=seg_parts).to(device)
    else:
        raise ValueError(f"Model {model_name} not implemented")

class JointTransform:
    """Applies the same spatial augmentations to both image and mask."""
    def __init__(self, augment=True, size=(300, 300)):
        self.augment = augment
        self.size = size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

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

class BreastDataset_seg(Dataset):
    """Dataset class for breast segmentation that matches the notebook expectations."""
    def __init__(self, time_point, data_dir, train_val="train", one_or_twoDim=False, denoise=False, seg_num=8, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train_val = train_val
        self.time_point = time_point
        self.one_or_twoDim = one_or_twoDim
        self.denoise = denoise
        self.seg_num = seg_num
        
        # Define file paths based on train_val split
        if train_val == "train":
            self.samples = list(range(1, 16))  # Example: Use samples 1-15 for training
        elif train_val == "val":
            self.samples = list(range(16, 21))  # Example: Use samples 16-20 for validation
        elif train_val == "inf":
            self.samples = list(range(21, 26))  # Example: Use samples 21-25 for inference
        else:
            raise ValueError("train_val must be 'train', 'val', or 'inf'")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample_idx = self.samples[idx]
        
        # Load DWI data
        dwi_path = os.path.join(self.data_dir, f'sample_{sample_idx}_NoisyDWIk.npy')
        dwi_data = np.load(dwi_path)
        
        # Load tissue label
        label_path = os.path.join(self.data_dir, f'sample_{sample_idx}_TissueType.npy')
        label_data = np.load(label_path)
        
        # Convert to tensor
        dwi_tensor = torch.from_numpy(dwi_data).float()
        label_tensor = torch.from_numpy(label_data).long()
        
        # Apply transform if available
        if self.transform is not None:
            dwi_tensor, label_tensor = self.transform(dwi_tensor, label_tensor)
            
        return dwi_tensor, label_tensor

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