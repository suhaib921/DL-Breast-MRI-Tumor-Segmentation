import numpy as np
import torch
from tqdm import tqdm
from unet.metrics import dice_coef, auc_roc, sensitivity, specificity, hausdorff_distance


def train_model(
    model, train_loader, val_loader, optimizer, loss_fn, epochs, device, save_path="best_model.pth"
):
    best_dice = 0.0
    model.to(device)
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
         # For tracking Dice on training set (optional)
        dice_sum = 0.0
        count = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation loop
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), save_path)
    
    return model, {"train_loss": train_loss, "val_metrics": val_metrics}

def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    dice, auc, sens, spec, hd = [], [], [], [], []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item() * images.size(0)
            
            # Compute metrics per batch
            outputs_np = outputs.cpu().numpy().squeeze()
            masks_np = masks.cpu().numpy().squeeze()
            
            for i in range(outputs_np.shape[0]):
                pred = outputs_np[i]
                true = masks_np[i]
                dice.append(dice_coef(pred, true))
                auc.append(auc_roc(pred, true))
                sens.append(sensitivity(pred, true))
                spec.append(specificity(pred, true))
                hd.append(hausdorff_distance(pred, true))
    
    # Aggregate metrics
    metrics = {
        "loss": total_loss / len(data_loader.dataset),
        "dice": np.nanmean(dice),
        "auc": np.nanmean(auc),
        "sensitivity": np.nanmean(sens),
        "specificity": np.nanmean(spec),
        "hausdorff": np.nanmean(hd),
    }
    return metrics

def train_seg(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, logger=None):
    """
    Training function for segmentation model that aligns with the notebook's expected function.
    """
    results = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_dice': []}
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        results['train_loss'].append(epoch_loss)
        
        # Validation phase every 2 epochs
        if epoch % 2 == 0:
            val_loss, val_acc, val_dice = check_accuracy(model, val_loader, criterion, device)
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_acc)
            results['val_dice'].append(val_dice)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
            
            if logger:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, '
                           f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}')
    
    return model, results

def check_accuracy(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)  # Ensure labels are float for BCE
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to probabilities and binary masks
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            # Calculate Dice for each sample in the batch
            for i in range(outputs.size(0)):
                pred = predicted[i].squeeze().cpu().numpy()
                true = labels[i].squeeze().cpu().numpy()
                dice = dice_coef(pred, true)
                dice_list.append(dice)
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_dice = np.nanmean(dice_list)
    return avg_loss, avg_dice

def calculate_multiclass_dice(pred, target, epsilon=1e-6):
    """
    Calculate Dice coefficient for multi-class segmentation.
    """
    # Flatten the tensors
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    # Calculate Dice coefficient for each class
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    # Return mean Dice over all classes
    return dice.mean().item()