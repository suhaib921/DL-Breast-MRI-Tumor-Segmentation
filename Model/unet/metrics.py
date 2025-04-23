import math
import torch
import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None  # Will be checked in hausdorff_distance function

def dice_coef(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between a predicted mask and ground truth mask.

    This function can take predicted probabilities and will threshold them to a binary mask (using the given threshold) 
    before computing the DSC. Dice coefficient is defined as:
      DSC = 2 * |X ∩ Y| / (|X| + |Y|),
    where X is the set of predicted positive pixels and Y is the set of true positive pixels.

    Args:
        y_pred: Predicted mask (as a NumPy array or torch Tensor). Can be probabilistic (float in [0,1]) or binary (0/1).
        y_true: Ground truth mask (NumPy array or torch Tensor), with binary values (0 or 1).
        threshold (float, optional): Threshold to binarize `y_pred` if it contains probabilities. Default is 0.5.

    Returns:
        float: Dice coefficient between 0.0 and 1.0. If both masks have no positive pixels, returns 1.0 (perfect overlap in trivial case).
    """
    # Convert inputs to NumPy arrays and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize the predicted mask
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    else:
        y_pred_bin = (y_pred != 0).astype(np.uint8)
    y_true_bin = (y_true != 0).astype(np.uint8)
    # Compute intersection and cardinalities
    intersection = np.sum(y_pred_bin * y_true_bin)
    pred_sum = np.sum(y_pred_bin)
    true_sum = np.sum(y_true_bin)
    # If both sets are empty (no positives in both prediction and truth), define Dice as 1.0
    if true_sum == 0 and pred_sum == 0:
        return 1.0
    # Compute Dice coefficient
    dice = 2.0 * intersection / (pred_sum + true_sum + 1e-8)
    return float(dice)

def auc_roc(y_pred, y_true) -> float:
    """
    Compute the Area Under the ROC Curve (AUC-ROC) for pixel-wise predictions.

    This treats each pixel's prediction as an independent probability of being positive, and computes the AUC-ROC 
    by comparing the distribution of prediction scores for positive vs negative pixels. The AUC-ROC is threshold-independent 
    and indicates the probability that a randomly chosen positive pixel has a higher predicted score than a randomly chosen negative pixel.

    Args:
        y_pred: Predicted probabilities for the positive class (NumPy array or torch Tensor of floats in [0,1]).
        y_true: Ground truth binary labels (NumPy array or torch Tensor of 0s and 1s).

    Returns:
        float: AUC-ROC value between 0.0 and 1.0. Returns NaN if `y_true` has all 0s or all 1s (AUC is undefined in those cases).
    """
    # Convert to NumPy and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(np.uint8)
    # Ensure ground truth is binary 0/1
    y_true = (y_true != 0).astype(np.uint8)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float('nan')  # AUC undefined if only one class present in truth
    # Compute AUC using the Mann-Whitney U statistic (equivalent to rank-based AUC calculation)
    # Sort predictions in increasing order and get sorted true labels
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    N = y_true_sorted.size
    # Compute the rank sum for positive labels
    rank_sum_pos = 0.0
    i = 0
    while i < N:
        # Handle ties in prediction scores by averaging their rank positions
        j = i
        while j < N and y_pred[order[i]] == y_pred[order[j]]:
            j += 1
        # Now indices [i, j) have the same score
        if j - i > 1:
            # If there are ties, assign average rank to all tied elements
            avg_rank = 0.5 * ((i + 1) + j)  # average of ranks (i+1) through j
            # Add total contribution of positives in this tied group
            rank_sum_pos += avg_rank * np.sum(y_true_sorted[i:j])
        else:
            # No tie, if this pixel is positive, add its rank (i+1, using 1-indexed rank)
            if y_true_sorted[i] == 1:
                rank_sum_pos += (i + 1)
        i = j
    # Compute AUC from rank sum
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2)) / float(n_pos * n_neg)
    return float(auc)

def sensitivity(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute sensitivity (True Positive Rate) for the segmentation.

    Sensitivity = TP / (TP + FN), where:
      - TP (True Positives) is the number of correctly predicted positive pixels (overlap of prediction and ground truth positives).
      - FN (False Negatives) is the number of missed positive pixels (ground truth positive but predicted negative).

    Args:
        y_pred: Predicted mask (probabilities or binary values).
        y_true: Ground truth binary mask.
        threshold (float, optional): Threshold to binarize `y_pred` if it contains probabilities. Default is 0.5.

    Returns:
        float: Sensitivity in the range [0, 1]. Returns NaN if there are no true positive pixels in `y_true` (no positives in ground truth).
    """
    # Convert to NumPy and flatten arrays
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize prediction
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold)
    else:
        y_pred_bin = (y_pred != 0)
    y_true_bin = (y_true != 0)
    # True positives (TP): prediction and truth both 1
    TP = np.sum(y_pred_bin & y_true_bin)
    # False negatives (FN): missed true positives (truth is 1, pred is 0)
    FN = np.sum((~y_pred_bin) & y_true_bin)
    # If no true positives in ground truth, sensitivity is undefined (return NaN)
    if (TP + FN) == 0:
        return float('nan')
    return float(TP / (TP + FN + 1e-8))

def specificity(y_pred, y_true, threshold: float = 0.5) -> float:
    """
    Compute specificity (True Negative Rate) for the segmentation.

    Specificity = TN / (TN + FP), where:
      - TN (True Negatives) is the number of correctly predicted background pixels (both prediction and ground truth are background).
      - FP (False Positives) is the number of pixels predicted as positive incorrectly (prediction is positive but ground truth is background).

    Args:
        y_pred: Predicted mask (probabilities or binary values).
        y_true: Ground truth binary mask.
        threshold (float, optional): Threshold to binarize `y_pred` if probabilistic. Default is 0.5.

    Returns:
        float: Specificity in the range [0, 1]. Returns NaN if there are no true negative pixels in ground truth (i.e., ground truth has no background).
    """
    # Convert to NumPy and flatten
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    # Binarize prediction
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold)
    else:
        y_pred_bin = (y_pred != 0)
    y_true_bin = (y_true != 0)
    # True negatives (TN): both pred and truth are 0
    TN = np.sum((~y_pred_bin) & (~y_true_bin))
    # False positives (FP): pred is 1 but truth is 0
    FP = np.sum(y_pred_bin & (~y_true_bin))
    # If ground truth has no background (no true negatives), specificity undefined
    if (TN + FP) == 0:
        return float('nan')
    return float(TN / (TN + FP + 1e-8))

def hausdorff_distance(y_pred, y_true, percentile: float = 95.0) -> float:
    """
    Compute the Hausdorff Distance between two binary masks, using a given percentile (HD95 by default).

    The Hausdorff distance is the maximum distance of any point on one mask to the nearest point on the other mask. 
    The percentile version (e.g., 95th percentile Hausdorff) is a robust variation that ignores a small fraction of outlier points.
    For example, the 95th percentile Hausdorff Distance (HD95) is the distance such that 95% of the points on one mask 
    are within that distance of the other mask's nearest point (and vice versa).

    This implementation uses the Euclidean distance in pixel units.

    Args:
        y_pred: Predicted mask (NumPy array or torch Tensor). If probabilities are provided, they will be thresholded at 0.5.
        y_true: Ground truth mask (NumPy array or torch Tensor).
        percentile (float, optional): Percentile for the Hausdorff distance. Default is 95.0 (for HD95). 
                                      Use 100.0 for the maximum Hausdorff distance.

    Returns:
        float: The Hausdorff distance at the given percentile. 
               Returns 0.0 if both masks have no foreground.
               Returns infinity if one mask has no foreground while the other has at least one (distance is unbounded in that case).
    """
    # Convert inputs to NumPy boolean arrays
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    # Binarize masks (treat values > 0.5 as True)
    y_pred_bin = (y_pred > 0.5)
    y_true_bin = (y_true > 0.5)
    # Edge cases: if both masks have no foreground, HD = 0; if only one has foreground, HD = inf
    pred_foreground = np.any(y_pred_bin)
    true_foreground = np.any(y_true_bin)
    if not pred_foreground and not true_foreground:
        return 0.0
    if not pred_foreground or not true_foreground:
        return float('inf')
    # Ensure SciPy is available for distance transform
    if distance_transform_edt is None:
        raise ImportError("SciPy is required for hausdorff_distance calculation.")
    # Compute distance transform of the complement (background) of each mask
    # dist_to_true[y, x] = distance from pixel (y,x) to the nearest foreground pixel in y_true_mask
    dist_to_true = distance_transform_edt(~y_true_bin)
    # dist_to_pred[y, x] = distance from pixel (y,x) to the nearest foreground pixel in y_pred_mask
    dist_to_pred = distance_transform_edt(~y_pred_bin)
    # For all foreground pixels in one mask, find distances to the other mask
    pred_to_true_dists = dist_to_true[y_pred_bin]  # distances from each pred foreground pixel to nearest true pixel
    true_to_pred_dists = dist_to_pred[y_true_bin]  # distances from each true foreground pixel to nearest pred pixel
    # Compute the given percentile of distances for both directions
    perc = float(percentile)
    hd_pred_to_true = np.percentile(pred_to_true_dists, perc)
    hd_true_to_pred = np.percentile(true_to_pred_dists, perc)
    # The Hausdorff distance at the percentile is the maximum of the two directed distances
    hd_percentile = max(hd_pred_to_true, hd_true_to_pred)
    return float(hd_percentile)
