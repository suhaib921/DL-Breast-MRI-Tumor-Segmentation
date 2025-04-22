import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None  # Will be checked in hausdorff_distance function

class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions (with padding) each followed by BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    """
    U-Net model for binary segmentation.

    This model implements the U-Net architecture&#8203;:contentReference[oaicite:1]{index=1} with an encoder-decoder structure and skip connections.
    It takes an input tensor of shape (N, C, H, W) and outputs a tensor of shape (N, 1, H, W) with a segmentation mask 
    for the positive class (e.g., tumor). All convolutions use padding to preserve spatial dimensions, so the output H, W 
    match the input. No pre-trained weights are used (the network is initialized randomly).

    Args:
        in_channels (int): Number of channels in the input images (e.g., 3 for RGB).
        out_channels (int): Number of output channels (for binary segmentation, use 1).
        init_features (int): Number of feature channels in the first convolution layer (default=64). This will be doubled at each downsampling step.

    Example:
        model = UNet(in_channels=3, out_channels=1, init_features=64)
        pred_mask = model(torch.randn(1, 3, 300, 300))  # pred_mask will have shape (1, 1, 300, 300)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 64):
        super(UNet, self).__init__()
        features = init_features
        # Encoder (contracting path)
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)
        # Decoder (expanding path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4   = DoubleConv(features * 16, features * 8)  # concat will double the channels
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3   = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2   = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1   = DoubleConv(features * 2, features)
        # Final 1x1 convolution (output)
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding path
        x1 = self.enc1(x)               # Shape: (N, features, H, W)
        x2 = self.enc2(self.pool1(x1))  # (N, features*2, H/2, W/2)
        x3 = self.enc3(self.pool2(x2))  # (N, features*4, H/4, W/4)
        x4 = self.enc4(self.pool3(x3))  # (N, features*8, H/8, W/8)
        x5 = self.bottleneck(self.pool4(x4))  # (N, features*16, H/16, W/16)
        # Decoding path with skip connections
        u4 = self.upconv4(x5)           # (N, features*8, H/8, W/8) after up-convolution
        # If spatial dimensions do not match due to pooling rounding, pad the upsampled feature map
        if x4.size(2) != u4.size(2) or x4.size(3) != u4.size(3):
            u4 = F.pad(u4, (0, x4.size(3) - u4.size(3), 0, x4.size(2) - u4.size(2)))
        x4 = torch.cat((x4, u4), dim=1)  # concat along channels
        x4 = self.dec4(x4)

        u3 = self.upconv3(x4)           # (N, features*4, H/4, W/4)
        if x3.size(2) != u3.size(2) or x3.size(3) != u3.size(3):
            u3 = F.pad(u3, (0, x3.size(3) - u3.size(3), 0, x3.size(2) - u3.size(2)))
        x3 = torch.cat((x3, u3), dim=1)
        x3 = self.dec3(x3)

        u2 = self.upconv2(x3)           # (N, features*2, H/2, W/2)
        if x2.size(2) != u2.size(2) or x2.size(3) != u2.size(3):
            u2 = F.pad(u2, (0, x2.size(3) - u2.size(3), 0, x2.size(2) - u2.size(2)))
        x2 = torch.cat((x2, u2), dim=1)
        x2 = self.dec2(x2)

        u1 = self.upconv1(x2)           # (N, features, H, W)
        if x1.size(2) != u1.size(2) or x1.size(3) != u1.size(3):
            u1 = F.pad(u1, (0, x1.size(3) - u1.size(3), 0, x1.size(2) - u1.size(2)))
        x1 = torch.cat((x1, u1), dim=1)
        x1 = self.dec1(x1)

        output = self.conv_out(x1)      # (N, 1, H, W)
        # Apply sigmoid activation to get probabilities in [0,1]
        output = torch.sigmoid(output)
        return output

def dice_coef(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between predicted and ground truth masks.

    This function converts the predicted probabilities to a binary mask (using the given threshold) 
    and then computes DSC = 2 * (|X ∩ Y|) / (|X| + |Y|), where X is the set of predicted positive pixels 
    and Y is the set of true positive pixels.

    Args:
        y_pred: Predicted mask (as a NumPy array or torch Tensor). Can be probabilities (float in [0,1]) or binary values.
        y_true: Ground truth mask (NumPy array or torch Tensor), with binary values (0 or 1).
        threshold (float, optional): Threshold to binarize `y_pred` if it is not already binary. Default is 0.5.

    Returns:
        float: Dice coefficient between 0.0 and 1.0. Returns 1.0 if both masks are completely empty (no positive pixels).
    """
    # Convert inputs to NumPy arrays
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize predictions
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    else:
        y_pred_bin = (y_pred != 0).astype(np.uint8)
    y_true_bin = (y_true != 0).astype(np.uint8)
    # Compute counts
    intersection = np.sum(y_pred_bin * y_true_bin)
    pred_sum = np.sum(y_pred_bin)
    true_sum = np.sum(y_true_bin)
    # If no foreground in either, define Dice as 1 (nothing to segment, and prediction matches truth trivially)
    if true_sum == 0 and pred_sum == 0:
        return 1.0
    # Compute Dice coefficient
    dice = 2.0 * intersection / (pred_sum + true_sum)
    return float(dice)

def sensitivity(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute sensitivity (True Positive Rate) for the segmentation.

    Sensitivity = TP / (TP + FN), where:
      - TP is the number of true positive pixels (predicted tumor pixels that match ground truth tumor),
      - FN is the number of false negative pixels (ground truth tumor pixels missed in prediction).

    Args:
        y_pred: Predicted mask (probabilities or binary).
        y_true: Ground truth binary mask.
        threshold (float, optional): Threshold to binarize `y_pred` if it's probabilistic. Default 0.5.

    Returns:
        float: Sensitivity (range 0 to 1). Returns NaN if there are no true positive pixels in ground truth (i.e., no tumor in GT).
    """
    # Convert to NumPy and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold)
    else:
        y_pred_bin = (y_pred != 0)
    y_true_bin = (y_true != 0)
    # Count positives
    TP = np.sum(y_pred_bin & y_true_bin)
    FN = np.sum((~y_pred_bin) & y_true_bin)
    if (TP + FN) == 0:
        return math.nan  # no positive pixels in ground truth to measure sensitivity
    return float(TP / (TP + FN))

def specificity(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute specificity (True Negative Rate) for the segmentation.

    Specificity = TN / (TN + FP), where:
      - TN is the number of true negative pixels (predicted background pixels that match ground truth background),
      - FP is the number of false positive pixels (predicted tumor pixels where ground truth is background).

    Args:
        y_pred: Predicted mask (probabilities or binary).
        y_true: Ground truth binary mask.
        threshold (float, optional): Threshold to binarize `y_pred` if probabilistic. Default 0.5.

    Returns:
        float: Specificity (range 0 to 1). Returns NaN if there are no true negative pixels in ground truth (i.e., ground truth is all tumor).
    """
    # Convert to NumPy and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold)
    else:
        y_pred_bin = (y_pred != 0)
    y_true_bin = (y_true != 0)
    # Count negatives
    TN = np.sum((~y_pred_bin) & (~y_true_bin))
    FP = np.sum(y_pred_bin & (~y_true_bin))
    if (TN + FP) == 0:
        return math.nan  # no negative pixels in ground truth to measure specificity
    return float(TN / (TN + FP))

def auc_roc(y_pred, y_true) -> float:
    """
    Compute the Area Under the ROC Curve (AUC-ROC) for pixel-wise predictions.

    This treats each pixel's prediction as a binary classification score. The AUC-ROC is threshold-independent 
    and indicates the probability that a randomly chosen positive pixel is ranked higher (has a higher predicted score) 
    than a randomly chosen negative pixel.

    Args:
        y_pred: Predicted probabilities for the positive class (NumPy array or torch Tensor of floats in [0,1]).
        y_true: Ground truth binary labels (NumPy array or torch Tensor of 0s and 1s).

    Returns:
        float: AUC-ROC value between 0.0 and 1.0. Returns NaN if `y_true` has all 0s or all 1s (AUC undefined in that case).
    """
    # Convert to NumPy and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(np.uint8)
    # Ensure binary ground truth
    y_true = (y_true != 0).astype(np.uint8)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return math.nan  # AUC is not defined if only one class present in truth
    # Compute AUC via rank ordering (Mann-Whitney U statistic)
    # Sort scores and corresponding true labels
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    # Compute rank sum for positive labels (with tie handling by average rank)
    N = y_true_sorted.size
    rank_sum_pos = 0.0
    i = 0
    while i < N:
        # Find range of tied scores
        j = i + 1
        while j < N and y_pred[order[j]] == y_pred[order[i]]:
            j += 1
        # All indices from i to j-1 have the same score
        if j - i > 1:
            # Average rank for tied scores = (i+1 + j) / 2 (1-based rank indexing)
            avg_rank = 0.5 * ((i + 1) + j)
            # Sum of ranks for positive labels in this tie group
            rank_sum_pos += avg_rank * np.sum(y_true_sorted[i:j])
        else:
            if y_true_sorted[i] == 1:
                rank_sum_pos += (i + 1)
        i = j
    # Mann-Whitney U statistic for positives
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2)) / float(n_pos * n_neg)
    return float(auc)

def hausdorff_distance(y_pred, y_true, percentile: float = 95.0) -> float:
    """
    Compute the Hausdorff distance between two binary masks, at a given percentile (HD percentile).

    The Hausdorff distance is the maximum distance from any point on one mask to the nearest point on the other mask.
    The percentile version (e.g., 95th percentile) computes a robust distance that ignores a small fraction of outlier points.
    Specifically, for 95th percentile Hausdorff (HD95), we take the 95th percentile of distances in both directions.

    This implementation uses Euclidean distance (in pixel units).

    Args:
        y_pred: Predicted binary mask (NumPy array or torch Tensor). If probabilities are provided, they will be thresholded at 0.5.
        y_true: Ground truth binary mask.
        percentile (float, optional): Percentile for the Hausdorff distance (default 95.0 for HD95). Use 100 for the absolute Hausdorff distance.

    Returns:
        float: Hausdorff distance at the given percentile. Returns 0.0 if both masks have no foreground.
               Returns math.inf if one mask has no foreground while the other has at least one (since the distance is unbounded).
    """
    # Convert to NumPy boolean arrays
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    # Binarize the masks (threshold at 0.5 for probabilities)
    y_pred_bin = (y_pred > 0.5)
    y_true_bin = (y_true > 0.5)
    # Edge cases: check for empty masks
    pred_foreground = np.any(y_pred_bin)
    true_foreground = np.any(y_true_bin)
    if not pred_foreground and not true_foreground:
        return 0.0
    if not pred_foreground or not true_foreground:
        return math.inf
    # Require SciPy for distance transform
    if distance_transform_edt is None:
        raise ImportError("SciPy is required for hausdorff_distance calculation.")
    # Compute distance transform for each mask's background
    # dist_to_true: for each pixel, distance to nearest true foreground pixel
    dist_to_true = distance_transform_edt(~y_true_bin)
    # dist_to_pred: for each pixel, distance to nearest predicted foreground pixel
    dist_to_pred = distance_transform_edt(~y_pred_bin)
    # Distances from each predicted foreground pixel to nearest true foreground pixel
    pred_to_true_dists = dist_to_true[y_pred_bin]
    # Distances from each true foreground pixel to nearest predicted foreground pixel
    true_to_pred_dists = dist_to_pred[y_true_bin]
    # Compute the given percentile of distances for each set
    perc = float(percentile)
    hd_pred_to_true = np.percentile(pred_to_true_dists, perc)
    hd_true_to_pred = np.percentile(true_to_pred_dists, perc)
    # Hausdorff distance at percentile = max of the two directed distances
    hd_percentile = max(hd_pred_to_true, hd_true_to_pred)
    return float(hd_percentile)
