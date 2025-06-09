### src/train.py
### Train Script
### Train a model on the Dataset
### Note on quotations: use '' for parameters, use "" for strings and prints.
import argparse
import os
import cv2 # fast BGR img I/O
import random, numpy as np, math
import torch
from torch import nn, optim, amp
from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingWarmRestarts, SequentialLR) # LR Scheduler
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchvision import transforms
# from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchmetrics.classification import MultilabelAUROC
import torch.nn.functional as TMF
from torch.multiprocessing import freeze_support
from tqdm.auto import tqdm # Progress Bar
import albumentations as ALB # aug lib
from albumentations.pytorch import ToTensorV2 # convert to torch.tensor

from dataset import ChestXRay14, LABELS

def main():
    ### Seed to reproduce results
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # GPU Check
    print("CUDA available :", torch.cuda.is_available())
    print("Visible devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print("Current device :", idx, torch.cuda.get_device_name(idx))
        print("Allocated / Reserved (MB):",
              torch.cuda.memory_allocated() // 2**20, "/",
              torch.cuda.memory_reserved()  // 2**20)

    # move up to project root
    #os.chdir("..")
    print("CMD Now: ", os.getcwd())

    # Parse for Colab
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/images",
                        help="folder that contains .png files")
    parser.add_argument("--csv_file", default="cxr_csv/Data_Entry_2017.csv")
    ARGS = parser.parse_args()
    ### Config
    IMG_DIR     = ARGS.img_dir  # "data/images" # full folder
    print("IMG_DIR resolved to:", IMG_DIR)
    CSV_FILE    = ARGS.csv_file  # "cxr_csv/Data_Entry_2017.csv"
    BATCH       = 16
    LR          = 1e-4
    EPOCHS      = 20
    patience_counter = 2
    # LR Scheduler
    warmup_epochs = 2
    first_restart = 4 # T_0 for Cosine...

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if Available

    ### Load Data
    #ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE)
    #n = len(ds)
    #n_train = int(n * 0.8) # train samples 80%

    # New way of Loading Data
    train_ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE, split='train', seed=SEED)
    val_ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE, split='valid', seed=SEED)

    # Define separate transforms
    train_tf = ALB.Compose([
        ALB.Resize(320, 320, interpolation=cv2.INTER_LINEAR),
        ALB.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        ALB.HorizontalFlip(p=0.5),
        ALB.ShiftScaleRotate(
            shift_limit=0.0, scale_limit=0.0, rotate_limit=7, p=0.3 # have no idea here!
        ),
        ALB.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_tf = ALB.Compose([
        ALB.Resize(320, 320, interpolation=cv2.INTER_LINEAR),
        ALB.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        ALB.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    print("Data Transformed") # Debug

    #train_ds, val_ds = random_split(ds, [n_train, n - n_train]) # split was old, we use patient ID
    train_ds.tf = train_tf
    val_ds.tf = val_tf

    # Compute per-class positive weights
    counts = train_ds.targets.sum(dim=0) # positives per class (14,)
    pos_weight = (len(train_ds) - counts) / counts # tensor, shape (14,)
    pos_weight = pos_weight.to(torch.float32) # required dtype

    # Weighted Random Sampler instead of Shuffling
    targets = train_ds.targets # Tensor N,14 of 0/1
    class_inv = 1.0 / targets.sum(dim=0) # positives per class
    img_w = (targets * class_inv).sum(dim=1) # bigger is rarer
    img_w = torch.where(img_w == 0, class_inv.min(), img_w) # no finding
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=img_w,
        num_samples=len(train_ds),
        replacement=True
    )

    # Loaders
    # For train_loader, we used sampler instead of shuffle=True for better randomization/balance
    train_loader = DataLoader(
        train_ds, batch_size=BATCH,
        sampler=train_sampler, num_workers=4,
        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    print("Data Loaded") # Debug

    ### Model
    # model = densenet121(weights=DenseNet121_Weights.DEFAULT) # load pretrained densenet121
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, len(LABELS))
    )

    # Details:
    # The last layer of a model is usually a classifier layer,
    # Here we replace that layer with a new one with our target number of classes,
    # (14 in this case), and we make sure the model is adapted to our dataset and targets.
    # model.classifier = nn.Linear(model.classifier.in_features, len(LABELS)) # dsnet121

    model = model.to(device)
    print("Model Created") # Debug

    # LOSS, OPTIMIZER, METRIC, LR Scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) # Binary Crossentropy with LogitsLoss (Binary CE + Sigmoid)

    optimizer = optim.AdamW(
        model.parameters(), lr=LR # old
        #{"params": model.features.parameters(), "lr": 1e-5},
        #{"params": model.classifier.parameters(), "lr": 1e-4}
        ) # AdamW Optimizer (ADAM + Decoupled Weight Decay)

    metric = MultilabelAUROC(num_labels=len(LABELS)).to(device)

    # Learning Rate Scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=first_restart, T_mult=2)

    #? Old LR Scheduler
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau( # 0.5 Factor and 2 Patience
    #    optimizer=optimizer, mode='max', factor=0.5, patience=patience_counter # No verbose parameter
    #)

    scheduler = SequentialLR(
        optimizer=optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    scaler = amp.GradScaler()

    print("Model Features Created") # Debug

    best_auroc = 0.0 # Best AUROC to save the model

    # RTX Optimization
    torch.set_float32_matmul_precision('high')

    ### Train and Validate
    for epoch in range(EPOCHS):
        model.train() # training mode
        loop = tqdm (train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", unit="batch") # progress bar
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda'):
                logits = model(imgs) # forward pass
                loss = criterion(logits, targets) # calc loss
            optimizer.zero_grad(set_to_none=True) # clear grad
            scaler.scale(loss).backward() # backpropagation with scale
            scaler.step(optimizer) # optimizer step
            scaler.update() # update scale
            loop.set_postfix(loss=loss.item()) # update progress bar with loss

        model.eval() # validation mode
        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", unit="batch") # progress bar
        preds_list, targets_list = [], []
        metric.reset()  # Reset metric state at start of validation
        with torch.no_grad():
            for imgs, targets in loop:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                preds = torch.sigmoid(logits)
                preds_list.append(preds)
                targets_list.append(targets)
        all_preds = torch.cat(preds_list) # Concatenate
        all_targets = torch.cat(targets_list).int()
        val_auroc = metric(all_preds.to(device), all_targets.to(device))
        scheduler.step()

        # Used for ReduceLROnPlateau
        # Step the LR scheduler after validation and Print the change
        #old_lr = optimizer.param_groups[0]['lr']
        #scheduler.step(val_auroc)
        #new_lr = optimizer.param_groups[0]['lr']
        #if new_lr != old_lr:
        #    print(f"EPOCH {epoch}: Learning rate changed from {old_lr:.2e} to {new_lr:.2e}")

        loop.set_postfix(auroc=val_auroc.item())
        print(f"EPOCH {epoch+1} - Val AUROC: {val_auroc:.4f}")

        ### Save the Model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print("Early Stopping Triggered")
                break

if __name__ == "__main__":
    freeze_support()
    main()
