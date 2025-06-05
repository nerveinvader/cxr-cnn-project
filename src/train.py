### src/train.py
### Train Script
### Train a model on the Dataset
### Note on quotations: use '' for parameters, use "" for strings and prints.
import argparse
import os
import random, numpy as np
import torch
from torch import nn, optim
# WeigthedRandomSample to balance batches
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchvision import transforms
# from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchmetrics.classification import MultilabelAUROC
import torch.nn.functional as TMF
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
    BATCH       = 48
    LR          = 1e-4
    EPOCHS      = 10
    patience_counter = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if Available

    ### Load Data
    ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE)
    n = len(ds)
    n_train = int(n * 0.8) # train samples 80%

    # To augment training, we use random flip + small rotations so the models see variety
    # Define separate transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    print("Data Transformed") # Debug

    train_ds, val_ds = random_split(ds, [n_train, n - n_train]) # split train | valid
    train_ds.dataset.tf = train_tf
    val_ds.dataset.tf = val_tf
    # Compute pos_weight tensor for each of the 14 labels
    all_targets = ds.targets # shape(N, 14)
    pos = all_targets.sum(dim=0)
    neg = len(ds) - pos
    pos_weight = (neg/pos).to(device)
    # Sampler for balanced batches
    train_targets = all_targets[train_ds.indices] # (n_train, 14) only for train_set
    sample_weights = (train_targets.cpu() * pos_weight.cpu()).sum(dim=1).numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Loaders
    # For train_loader, we used sampler instead of shuffle=True for better randomization/balance
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    print("Data Loaded") # Debug

    ### Model
    # model = densenet121(weights=DenseNet121_Weights.DEFAULT) # load pretrained densenet121
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1) # new model effnetb0
    # Details:
    # The last layer of a model is usually a classifier layer,
    # Here we replace that layer with a new one with our target number of classes,
    # (14 in this case), and we make sure the model is adapted to our dataset and targets.
    # model.classifier = nn.Linear(model.classifier.in_features, len(ds.targets[0])) # dsnet121
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 14) # effnetb0
    model = model.to(device)

    print("Model Created") # Debug

    #* LOSS, OPTIMIZER, METRIC, LR Scheduler
    # criterion = nn.BCEWithLogitsLoss() # Binary Crossentropy with LogitsLoss (Binary CE + Sigmoid) # Old criterion
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Redefine criterion with class weights

    optimizer = optim.AdamW([
        # model.parameters(), lr=LR # old
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
        ]) # AdamW Optimizer (ADAM + Decoupled Weight Decay)

    metric = MultilabelAUROC(num_labels=len(ds.targets[0])).to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau( # 0.5 Factor and 2 Patience
        optimizer=optimizer, mode='max', factor=0.5, patience=patience_counter # No verbose parameter
    )

    print("Model Features Created") # Debug

    best_auroc = 0.0 # Best AUROC to save the model

    ### Train and Validate
    for epoch in range(EPOCHS):
        model.train() # training mode
        loop = tqdm (train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", unit="batch") # progress bar
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs) # forward pass
            loss = improved_focal_loss(logits, targets, alpha=0.25, gamma=2.0) # criterion(logits, targets) # calc loss - old
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
                preds = torch.sigmoid(logits)
                preds_list.append(preds)
                targets_list.append(targets)
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print("Early Stopping Triggered")
                break

def improved_focal_loss(logits, targets, pos_weight, alpha=0.25, gamma=2.0):
    bce = TMF.binary_cross_entropy_with_logits(
        logits,targets,
        reduction="none",
        reduce="none",
        pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = bce * ((1 - p_t + 1e-8) ** gamma)

    if alpha:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()

if __name__ == "__main__":
    freeze_support()
    main()
