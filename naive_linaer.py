import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mesh_dataset import MeshDataset,custom_collate_fn  # should now return (features, label)
from tools.helper import save_checkpoint, load_checkpoint, init_opt
from tqdm import tqdm
import os
from loss import MaskedCrossEntropyLoss
from torchmetrics import F1Score, Accuracy  # Additional metrics
from torchmetrics import ConfusionMatrix
# ----------------------------------------
# Configs
# ----------------------------------------
mesh_dir        = r'/bigwork/nhgnheid/H3D_orderdL/processed_meshes'
label_dir       = r'/bigwork/nhgnheid/H3D_orderdL/processed_labels'
checkpoint_dir  = r'/bigwork/nhgnheid/Ex_meja_downsteram/Ex_1'

epochs          = 100
batch_size      = 1
lr              = 0.001
resume          = False

input_dim       = 16
num_classes     = 12  # class_dim
N_class = 12
n_clusters= 300
PE= True
clusters_per_batch= 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------
# Dataset & DataLoader
# ----------------------------------------
# MeshDataset must now return (features: Tensor[B, input_dim], labels: LongTensor[B])
dataset = MeshDataset(mesh_dir, label_dir, n_clusters, clusters_per_batch, PE)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# ----------------------------------------
# Model, Loss, Optimizer
# ----------------------------------------
model     = nn.Linear(input_dim, num_classes).to(device)
criterion = MaskedCrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# (optional) scheduler, etc:
# scheduler = ...



# ----------------------------------------
# Checkpoint resume
# ----------------------------------------
start_epoch = 0
best_loss   = float('inf')
ckpt_path   = os.path.join(checkpoint_dir, 'best_model.pth')
if resume and os.path.exists(ckpt_path):
    start_epoch, best_loss = load_checkpoint(
        model, optimizer, scheduler=None, 
        checkpoint_path=ckpt_path, 
        iterations_per_epoch=len(data_loader)
    )
    print(f"> Resuming from epoch {start_epoch}, best loss={best_loss:.4f}")

# ----------------------------------------
# Training loop
# Initialize Ignite metrics
confusion_matrix = ConfusionMatrix(num_classes=N_class)

# Initialize torchmetrics metrics and move them to the correct device
f1_metric = F1Score(task='multiclass', num_classes=N_class, average='none', ignore_index=11).to(device)
accuracy_metric = Accuracy(task='multiclass', num_classes=N_class, average='none', ignore_index=11).to(device)
# ----------------------------------------
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0

    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for i, (features, labels, masks) in enumerate(data_loader):
            features = features.to(device)    # [B, input_dim]
            labels   = labels.to(device)      # [B]
            print('features',features.shape)
            print('labels',labels.shape)
            print('masks',masks.shape)
            logits = model(features.to(dtype=torch.float32))
            print('logits',logits.shape)
          # [B, num_classes]
            loss   = criterion(logits, labels,masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('running_loss',running_loss)
            pred = logits.reshape(-1, N_class)
            target = labels.reshape(-1)
            mask = masks.view(-1)
            valid_mask = mask == 1  # assuming mask is binary with 1 for valid data points
            pred = pred[valid_mask]
            target = target[valid_mask]
            # Update confusion matrix and IoU metric based on initial output
            confusion_matrix.update((pred, target))
            # Update torchmetrics
            f1_metric.update(pred, target)
            f1_scores = f1_metric.compute()
            mean_f1_score = f1_scores[:11].mean().item()
            accuracy_metric.update(pred, target)
            accuracy = accuracy_metric.compute()
            m_accuracy = accuracy[:11].mean().item()
            f1_scores = f1_metric.compute()
            mean_f1_score = f1_scores[:11].mean().item()
            accuracy = accuracy_metric.compute()
            m_accuracy = accuracy[:11].mean().item()
            print(f"Epoch {epoch + 1}, f1_scores: {f1_scores}, Mean F1 Score: {mean_f1_score}, Accuracy: {accuracy}, Mean accuracy: {m_accuracy}, miou: {miou}")
            # Evaluate on validation set


    epoch_loss = running_loss / len(data_loader)
    print(f"→ Epoch {epoch+1} complete: avg loss = {epoch_loss:.4f}")




# ----------------------------------------

print("Training finished.")
