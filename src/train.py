### src/train.py
### Train Script
### Train a model on the Dataset
### Note on quotations: use '' for parameters, use "" for strings and prints.

# Important Basic
import os
import random, numpy as np
import argparse # For parsing directories/files
import cv2 # fast BGR img I/O
# Torch
import torch
from torch import nn, optim, amp
# Learning Rate
from torch.optim.lr_scheduler import OneCycleLR ## New LR cycle JUNE 9
# Data Loading
from torch.utils.data import DataLoader
# Model
import timm # New backbone JUNE 9
from torchvision.models import densenet121, DenseNet121_Weights
import albumentations as ALB # aug lib
from albumentations.pytorch import ToTensorV2 # convert to torch.tensor
# Metrics
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
# Misc
from tqdm.auto import tqdm # Progress Bar
from torch.multiprocessing import freeze_support

from dataset import ChestXRay14, LABELS

class AsymmetricLoss(nn.Module): # JUNE 9
	"""
	Asymmetric Loss for multilabel classification,
	focuses hard negatives and ignores easy positives
	"""
	def __init__(self, rho, gamma_pos=1, gamma_neg=4.0, margin=0.05, eps=1e-8):
		super().__init__()
		self.gamma_pos, self.gamma_neg, self.eps = gamma_pos, gamma_neg, eps
		self.margin = margin
		self.register_buffer('weight_positive', torch.exp(1 - rho))
		self.register_buffer('weight_negative', torch.exp(rho))

	def forward(self, logits, targets):
		probs = torch.sigmoid(logits)
		probs_m = torch.clamp(probs - self.margin, min=0)

		loss_pos = targets * torch.log(probs.clamp(min=self.eps))
		loss_pos = loss_pos * (1 - probs) ** self.gamma_pos

		loss_neg = (1 - targets) * torch.log(1 - probs_m).clamp(min=self.eps)
		loss_neg = loss_neg * probs_m ** self.gamma_neg

		weights = self.weight_positive * targets + self.weight_negative * (1 - targets)
		loss = -(weights * (loss_pos + loss_neg))

		return loss.mean()

class EMA: # JUNE 9
	"""
	Exponential Moving Average of model parametes
	"""
	def __init__(self, model, decay=0.9995):
		self.decay = decay
		self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

	@torch.no_grad()
	def update(self, model):
		for n, p in model.named_parameters():
			if n in self.shadow:
				self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

	@torch.no_grad()
	def copy_to(self, model):
		for n, p in model.named_parameters():
			if n in self.shadow:
				p.data.copy_(self.shadow[n])

def main():
	### Seed to reproduce results
	SEED = 1996
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	# GPU Check
	# print("CUDA available :", torch.cuda.is_available())
	# print("Visible devices:", torch.cuda.device_count())
	# if torch.cuda.is_available():
	#     idx = torch.cuda.current_device()
	#     print("Current device :", idx, torch.cuda.get_device_name(idx))
	#     print("Allocated / Reserved (MB):",
	#           torch.cuda.memory_allocated() // 2**20, "/",
	#           torch.cuda.memory_reserved()  // 2**20)

	# move up to project root
	print("CMD Now: ", os.getcwd())

	# Parse for Colab
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_dir", default="data/images",
						help="folder that contains .png files")
	parser.add_argument("--csv_file", default="cxr_csv/Data_Entry_2017.csv")
	ARGS = parser.parse_args()

	#* Config
	IMG_DIR     = ARGS.img_dir  # "data/images" # full folder
	print("IMG_DIR resolved to:", IMG_DIR)
	CSV_FILE    = ARGS.csv_file  # "cxr_csv/Data_Entry_2017.csv"

	IMG_SIZE	= 224 # higher resolution JUNE 9
	BATCH       = 16
	LR          = 1e-3 # max_lr for one cycle JUNE 9
	EPOCHS      = 20
	patience_counter = 0

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if Available

	#* Load Data
	#ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE)
	#n = len(ds)
	#n_train = int(n * 0.8) # train samples 80%
	# New way of Loading Data
	train_ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE, split='train', seed=SEED)
	val_ds = ChestXRay14(img_dir=IMG_DIR, csv_file=CSV_FILE, split='valid', seed=SEED)

	#* Define separate transforms
	train_tf = ALB.Compose([
		ALB.Resize(height=IMG_SIZE, width=IMG_SIZE,p=1.0),
		# ALB.RandomResizedCrop([IMG_SIZE, IMG_SIZE], # JUNE 9
		#                       scale=(0.8, 1.0),
		#                       ratio=(0.9, 1.1),
		#                       p=1.0),
		ALB.HorizontalFlip(p=0.5),
		ALB.RandomBrightnessContrast(p=0.3), # JUNE 9
		ALB.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # JUNE 9
		ALB.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
		ALB.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]),
		ToTensorV2(),
	])
	val_tf = ALB.Compose([
		ALB.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_LINEAR),
		ALB.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
		ALB.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]),
		ToTensorV2(),
	])
	print("Data Transformed") # Debug

	train_ds.tf, val_ds.tf = train_tf, val_tf # transform

	# Compute per-class positive weights
	counts = train_ds.targets.sum(dim=0) # positives per class (14,)
	pos_weight = (len(train_ds) - counts) / counts # tensor, shape (14,)
	pos_weight = pos_weight.to(torch.float32) # required dtype

	#* Loaders
	# For train_loader, we used sampler instead of shuffle=True for better randomization/balance
	train_loader = DataLoader(
		train_ds, batch_size=BATCH,
		shuffle=True, num_workers=4,
		pin_memory=True, persistent_workers=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

	print("Data Loaded") # Debug

	#* Model
	model = densenet121(weights=DenseNet121_Weights.DEFAULT)
	model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))

	# Finalize Model
	model = model.to(device)
	print("Model Created") # Debug

	#* LOSS, OPTIMIZER, METRIC, LR Scheduler
	counts = train_ds.targets.sum(dim=0)
	rho = counts / len(train_ds)
	criterion = AsymmetricLoss(rho=rho.to(device)) # JUNE 9
	optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2) # AdamW Optimizer (ADAM + Decoupled Weight Decay)
	steps_per_epoch = len(train_loader)
	#* Learning Rate Scheduler
	scheduler = OneCycleLR( # JUNE 9
		optimizer=optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
		pct_start=0.15, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4
	)

	# Old LR Schedulers
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau( # 0.5 Factor and 2 Patience
	#    optimizer=optimizer, mode='max', factor=0.5, patience=patience_counter # No verbose parameter
	#)

	scaler = amp.GradScaler()
	auroc_metric = MultilabelAUROC(num_labels=len(LABELS), average=None).to(device)
	f1_metric = MultilabelF1Score(num_labels=len(LABELS), threshold=0.5, average=None).to(device)
	ema = EMA(model) # JUNE 9
	print("Model Features Created") # Debug

	best_auroc = 0.0 # Best AUROC to save the model

	#* RTX Optimization
	torch.set_float32_matmul_precision('high')

	### Train and Validate
	for epoch in range(EPOCHS):
		###* TRAIN
		model.train() # training mode
		loop = tqdm (train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", unit="batch") # progress bar
		for imgs, targets in loop:
			imgs, targets = imgs.to(device), targets.to(device)
			with torch.amp.autocast(device_type='cuda'):
				logits = model(imgs) # forward pass
				loss = criterion(logits, targets) # calc loss
			optimizer.zero_grad(set_to_none=True) # clear grad
			scaler.scale(loss).backward() # backpropagation with scale
			# JUNE 9
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping before stepping
			#
			scaler.step(optimizer) # optimizer step
			scaler.update() # update scale
			scheduler.step() # JUNE 9 One cycle per step
			ema.update(model) # JUNE 9
			loop.set_postfix(loss=loss.item()) # update progress bar with loss

		###* VALIDATE
		model.eval() # validation mode
		auroc_metric.reset()  # Reset metric state at start of validation
		f1_metric.reset()
		loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", unit="batch") # progress bar
		preds_list, targets_list = [], []
		with torch.no_grad():
			for imgs, targets in loop:
				imgs, targets = imgs.to(device), targets.to(device)
				# JUNE 9
				with torch.amp.autocast(device_type='cuda'):
					logits1 = model(imgs)
					logits2 = model(torch.flip(imgs, dims=[3]))
					logits = (logits1 + logits2) / 2.0
				#
				preds = torch.sigmoid(logits)
				auroc_metric.update(preds, targets.int()) # JUNE 9
				f1_metric.update(preds, targets.int())
				preds_list.append(preds)
				targets_list.append(targets)
		all_preds = torch.cat(preds_list) # Concatenate
		all_targets = torch.cat(targets_list).int()
		scheduler.step()

		# Used for ReduceLROnPlateau
		# Step the LR scheduler after validation and Print the change
		#old_lr = optimizer.param_groups[0]['lr']
		#scheduler.step(val_auroc)
		#new_lr = optimizer.param_groups[0]['lr']
		#if new_lr != old_lr:
		#    print(f"EPOCH {epoch}: Learning rate changed from {old_lr:.2e} to {new_lr:.2e}")

		auroc_all = auroc_metric.compute() # shape (14,)
		val_auroc = auroc_all.mean().item() # JUNE 9
		val_f1 = f1_metric.compute().mean().item()
		loop.set_postfix(auroc=val_auroc, f1=val_f1)
		print(f"EPOCH {epoch+1} - Val AUROC: {val_auroc:.4f} - Val F1: {val_f1:.4f}")
		if auroc_all.ndim:
			print("AUROC Per Label: ", [round(float(x),4) for x in auroc_all])
		else:
			print("AUROC Per Label: ", round(float(auroc_all.item()),4))

		### Save the Model
		if val_auroc > best_auroc:
			best_auroc = val_auroc
			ema.copy_to(model) # JUNE 9
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
