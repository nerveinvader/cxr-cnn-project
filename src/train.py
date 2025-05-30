### src/train.py
### Train Script
### Train a model on the Dataset
### Note on quotations: use '' for parameters, use "" for strings and prints.

import os
import random, numpy as np
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torchmetrics.classification import MultilabelAUROC
from torch.multiprocessing import freeze_support
from tqdm.auto import tqdm # Progress Bar

from dataset import ChestXRay14

def main():
    ### Seed to reproduce results
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # move up to project root
    #os.chdir("..")
    print("CMD Now: ", os.getcwd())

    ### Config
    IMG_DIR     = "data/images" # full folder
    CSV_FILE    = "cxr_csv/Data_Entry_2017.csv"
    BATCH       = 16
    LR          = 1e-4
    EPOCHS      = 5 # default was 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if Available

    ### Load Data
    ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE)
    n = len(ds)
    n_train = int(n * 0.8) # train samples 80%
    train_ds, val_ds = random_split(ds, [n_train, n - n_train]) # split train | valid
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    ### Model
    model = densenet121(weights=DenseNet121_Weights.DEFAULT) # load pretrained densenet121
    # Details:
    # The last layer of a model is usually a classifier layer,
    # Here we replace that layer with a new one with our target number of classes,
    # (14 in this case), and we make sure the model is adapted to our dataset and targets.
    model.classifier = nn.Linear(model.classifier.in_features, len(ds.targets[0]))
    model = model.to(device)

    ### LOSS, OPTIMIZER, METRIC, LR Scheduler
    criterion = nn.BCEWithLogitsLoss() # Binary Crossentropy with LogitsLoss (Binary CE + Sigmoid)
    optimizer = optim.AdamW(model.parameters(), lr=LR) # AdamW Optimizer (ADAM + Decoupled Weight Decay)
    metric = MultilabelAUROC(num_labels=len(ds.targets[0])).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( # 0.5 Factor and 2 Patience
        optimizer=optimizer, mode='max', factor=0.5, patience=2 # No verbose parameter
    )
    best_auroc = 0.0 # Best AUROC to save the model

    ### Train and Validate
    for epoch in range(EPOCHS):
        model.train() # training mode
        loop = tqdm (train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", unit="batch") # progress bar
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs) # forward pass
            loss = criterion(logits, targets) # calc loss
            optimizer.zero_grad() # reset grad ???
            loss.backward() # backpropagation
            optimizer.step() # update
            loop.set_postfix(loss=loss.item()) # update progress bar with loss

        model.eval() # validation mode
        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", unit="batch") # progress bar
        preds_list, targets_list = [], []
        metric.reset()  # Reset metric state at start of validation
        with torch.no_grad():
            for imgs, targets in loop:
                imgs = imgs.to(device)
                targets = targets.to(device)
                logits = model(imgs)
                preds = torch.sigmoid(logits).cpu()
                preds_list.append(preds)
                targets_list.append(targets.cpu())
        all_preds = torch.cat(preds_list) # Concatenate
        all_targets = torch.cat(targets_list).int()
        val_auroc = metric(all_preds.to(device), all_targets.to(device))
        # Step the LR scheduler after validation and Print the change
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auroc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"EPOCH {epoch}: Learning rate changed from {old_lr:.2e} to {new_lr:.2e}")

        loop.set_postfix(auroc=val_auroc.item())
        print(f"EPOCH {epoch+1} - Val AUROC: {val_auroc:.4f}")

        ### Save the Model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    freeze_support()
    main()
