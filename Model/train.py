import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import dice_loss, multiclass_dice_coeff, dice_coeff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm

from datetime import datetime
# def train(model, train_loader, loss_fn, optimizer, device, epochs, save_dir, load_path=None, save_interval=1, log_path=None, overfit_batch=False):

def run_batch(model, data, target, criterion, optimizer, device, is_training=True):

    # ---CODE START HERE---
    if is_training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    data = data.float().to(device)
    target = target.long().to(device).squeeze()

    # print("target",target.shape)

    output = model(data)
    # print("output",output.shape)
    temp = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float()
    # target = target.to(torch.long)
#     print(output.dtype, target.dtype)
    # print(output.shape, target.shape)
    loss = criterion(output, target)
    diceloss = dice_loss(F.softmax(output, dim=1).float(),
                                                    F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float(),
                                                    multiclass=True)
    
    

    if is_training:
        loss.backward()
        optimizer.step()

    # ---CODE END HERE---

    return loss.item()

def check_accuracy(model, loader, criterion, device):
    """
    Check accuracy of our trained model given a loader and a model

    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on

    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    total_loss = 0.0
    num_batches = 0
    dice_vali_list = []
    dice_vali_all_list = []

    accuracy_vali_list = []
    precision_vali_list = []
    f1_vali_list = []
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        print(type(loader))
        for x, y in tqdm(loader):

            # Move data to device
            x = x.float().to(device)
            # print(y.shape)
            y = y.long().to(device)

            # Get to correct shape

            # Forward pass
            scores = model(x)
            mask_true = F.one_hot(y, model.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(scores.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
            # Compute loss
  
            loss = criterion(scores, y.long().to(device))
            
            # calculate metrices
            # print(scores.shape,"shape")
            pred = scores.argmax(dim=1).cpu().numpy()
            gt = y.cpu().numpy()
            # print(pred.shape, y.shape)#ok, 1,200,200
            # print("min", pred.max(), y.max()) #ok, 7
            accuracy = accuracy_score(pred.flatten(), gt.flatten())
            precision = precision_score(pred.flatten(), gt.flatten(), average='weighted')
            # recall = recall_score(pred.flatten(), gt.flatten(), average='weighted')
            f1 = f1_score(pred.flatten(), gt.flatten(), average='weighted')

            accuracy_vali_list.append(accuracy)
            precision_vali_list.append(precision)
            f1_vali_list.append(f1)


            dicescore_vali = multiclass_dice_coeff(mask_true, mask_pred, reduce_batch_first=False)
            
            dice_vali_list.append(dicescore_vali[0].item())
            dice_vali_all_list.append(dicescore_vali[1].cpu().numpy())
            
            # print(dice_vali_list)
            # Check how many we got correct
            # num_correct += (predictions.reshape(-1) == y.reshape(-1)).sum().item()
            total_loss += loss.item()
            # Keep track of number of samples
            # num_samples += predictions.shape[0]
            num_batches += 1
    model.train()
    print("dice", np.mean(dice_vali_list))
    print("accuracy",np.mean(accuracy_vali_list))
    print("precision",np.mean(precision_vali_list))
    print("f1",np.mean(f1_vali_list))

    return total_loss / num_batches, np.mean(dice_vali_list), np.mean(dice_vali_all_list, axis=0)

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Complete the code between ---CODE START HERE--- and ---CODE END HERE---:
    1. Move data and target to the specified device
    2. Call run_batch with appropriate parameters
    """
    model.train()
    total_loss = 0.0

    for data, target in tqdm(loader):
        # ---CODE START HERE---
        data, target = data.to(device), target.to(device)
        loss = run_batch(model, data, target, criterion, optimizer, device=device)
        total_loss += loss
        # ---CODE END HERE---

    avg_loss = total_loss / len(loader)
    return avg_loss

def train_seg(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, check_every=2, logger = None):
    """
    Train the model for multiple epochs.

    Complete the code between ---CODE START HERE--- and ---CODE END HERE---:
    1. Call train_epoch with appropriate parameters
    2. Store the returned train_loss and train_acc
    """
    current_time = datetime.now().strftime("%H_%M_%S")
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'val_accuracy': []
    }
    
    best_loss = np.inf

    for epoch in range(num_epochs):
        # ---CODE START HERE---
        print(f"==epoch {epoch+1} start==")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # ---CODE END HERE---

        results['train_loss'].append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}:")

        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        if epoch % check_every == 0:
            val_loss, val_accuracy, val_all = check_accuracy(model, val_loader, criterion, device)
            print("vall", val_all)
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_accuracy)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            logger.info('loss: {:.4f} dice: {:.4f}'.format(val_loss, val_accuracy))

            # save
            if val_loss < best_loss:
                
                torch.save(model.state_dict(), f'model_weights_{len(val_all)}_{current_time}.pth')


    return model, results