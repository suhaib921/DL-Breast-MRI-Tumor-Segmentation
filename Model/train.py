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
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
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