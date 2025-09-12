import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mesh_dataset_j_4 import MeshDataset, custom_collate_fn 
# from model import nomeformer  
import time
import matplotlib.pyplot as plt  
from model_G_2 import nomeformer , NomeformerPredictor
from loss import IJEPALoss  
from ignite.metrics import IoU, ConfusionMatrix
from torchmetrics import F1Score, Accuracy
from  tools.helper import save_checkpoint , load_checkpoint
from tqdm import tqdm
import os
import copy
from tools.helper import init_opt
import math
import torch.nn.functional as F
import numpy as np
import random
import trimesh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Directory containing mesh files
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model with a given configuration file.')
parser.add_argument('config_path', type=str, help='Path to the configuration file.')
args = parser.parse_args()

# Load configuration
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

# Accessing configurations
mesh_dir = config['paths']['mesh_dir']
log_dir = config['paths']['log_dir']
checkpoint_dir = config['paths']['checkpoint_dir']

# Get masking parameters from config
masking_ratio_range = config['masking']['ratio_range']  # e.g., [0.3, 0.7]
num_mask_blocks     = config['masking'].get('num_mask_blocks', 1)  # default to 1 if not set
clusters_to_select = config['masking']['clusters_to_select']  # e.g., 10

Training_epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
clusters_per_batch = config['training']['clusters_per_batch']
learning_rate = config['training']['learning_rate']
accumulation_steps = config['training']['accumulation_steps']
resume = config['training']['resume']
optimize_target = config['training'].get('optimize_target_encoder', False)
n_clusters = config['model']['n_clusters']
feature_dim = config['model']['feature_dim']
embedding_dim = config['model']['embedding_dim']
num_heads = config['model']['num_heads']
num_attention_blocks = config['model']['num_attention_blocks']
num_attention_blocks_perdictor = config['model']['num_attention_blocks_perdictor'] 
n_clusters = config['model']['n_clusters']
faces_per_cluster = config['model']['faces_per_cluster']
PE = config['model']['use_pe']
Gradinat_ac = config['model']['gradinat_ac']
dropout = config['model'].get('dropout', 0.1) 
use_hierarchical = config['model'].get('use_hierarchical', False)

# Load loss weights from config, defaulting to 1.0 if not present

alpha = config['loss'].get('alpha', 1.0)
beta = config['loss'].get('beta', 0.0)
gamma = config['loss'].get('gamma', 0.0)
def compute_masked_std(context_enc, predictor, target_enc, target_masks):
    """
    Compute the mean-over-dimension std only over positions where target_masks is True.

    Args:
        context_enc   (Tensor): [B, N, D] context embeddings (unused in this example, but available if you want it)
        predictor     (Tensor): [B, N, D] model predictions
        target_enc    (Tensor): [B, N, D] target embeddings
        target_masks  (Tensor): [B, N] boolean mask

    Returns:
        std_predictor (float): mean of std across D for masked predictor rows
        std_target    (float): mean of std across D for masked target rows
    """
    # flatten to [B*N, D]
    flat_pred   = predictor.view(-1, predictor.size(-1))
    flat_tgt    = target_enc.view(-1,    target_enc.size(-1))
    flat_mask   = target_masks.view(-1).bool()  # [B*N]

    # select only masked rows
    masked_pred = flat_pred[flat_mask]
    masked_tgt  = flat_tgt[flat_mask]

    # compute per-dimension std, then average
    std_pred = masked_pred.std(dim=0).mean().item()
    std_tgt  = masked_tgt.std(dim=0).mean().item()
    return std_pred, std_tgt

mode = config['augmentation']['mode']
optimizer_type = None
if 'optimizer' in config:
    # Access optimizer settings from the config
    optimizer_config = config['optimizer']
    optimizer_type = optimizer_config['type']
    learning_rate = optimizer_config['learning_rate']
    warmup = optimizer_config['warmup']
    start_lr = optimizer_config['start_lr']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate dataset and DataLoader
dataset = MeshDataset(
    mesh_dir=mesh_dir,
    clusters_per_batch=clusters_per_batch,
    faces_per_cluster=faces_per_cluster,
    PE=PE,
    augmentation=None,
    flexible_num_clusters=False,
    n_clusters=n_clusters,
)
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,
    collate_fn=lambda batch: custom_collate_fn(
        batch, 
        masking_ratio_range=masking_ratio_range,
        clusters_to_select=clusters_to_select,
        num_pred_blocks = num_mask_blocks
        
    )
)
iterations_per_epoch = len(data_loader)
print('iterations_per_epoch',iterations_per_epoch)
# model = nomeformer(feature_dim, embedding_dim, num_heads, num_attention_blocks, N_class)
# context_encoder = nomeformer(feature_dim, embedding_dim, num_heads, num_attention_blocks, dropout).to(device)
context_encoder = nomeformer(
    feature_dim=feature_dim,            # your input feature size
    embedding_dim=embedding_dim,
    num_heads=2,
    num_attention_blocks=num_attention_blocks,
    dropout=dropout,
    summary_mode='cls',       # or 'avg'
    use_hierarchical=use_hierarchical,    # or False
    num_hierarchical_stages=1,
    fourier=False,             # keep as you use now
    relative_positional_encoding=False  # <-- activate RPE
).to(device)
context_predictor = NomeformerPredictor(feature_dim, embedding_dim, num_heads, num_attention_blocks_perdictor, dropout).to(device)
# Create the target encoder as a deep copy of the model
target_encoder = copy.deepcopy(context_encoder).to(device)

# Conditionally freeze (or not) the target‐encoder
if optimize_target:
    # train target_encoder alongside context_encoder & predictor
    print("ℹ️  Training target_encoder with gradients")
    for param in target_encoder.parameters():
        param.requires_grad = True
else:
    # MoCo/BYOL style: freeze and momentum‐update
    print("ℹ️  Freezing target_encoder; will update via momentum")
    for param in target_encoder.parameters():
        param.requires_grad = False

# Define momentum coefficient (similar to MoCo/BYOL)
base_momentum = 0.996


def update_target_encoder(model, target_encoder, momentum=base_momentum ):
    """Momentum update of the target encoder parameters."""
    with torch.no_grad():
        for param_q, param_k in zip(model.parameters(), target_encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.data)
#print(f"Number of parameters in the model: {num_params}")
print('Training_epochs',Training_epochs)

# ===== PCA coloring utilities =====
def _pca_to_rgb(features_np):
    scaler = StandardScaler()
    X = scaler.fit_transform(features_np)
    pcs3 = PCA(n_components=3, random_state=0).fit_transform(X)
    mins = pcs3.min(axis=0, keepdims=True)
    maxs = pcs3.max(axis=0, keepdims=True)
    denom = np.clip(maxs - mins, 1e-8, None)
    pcs01 = (pcs3 - mins) / denom
    colors = (pcs01 * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.concatenate([colors, 255*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
    return rgba

@torch.no_grad()
def visualize_random_mesh_pca(target_encoder, mesh_dir, save_root, device, faces_per_cluster, clusters_per_batch, use_pe: bool, n_clusters_fixed: int = None):
    os.makedirs(save_root, exist_ok=True)
    # choose a random mesh file
    candidates = [f for f in os.listdir(mesh_dir) if f.endswith(('.obj', '.ply', '.off'))]
    if not candidates:
        print('No mesh files found for PCA visualization.')
        return
    fname = random.choice(candidates)
    mesh_path = os.path.join(mesh_dir, fname)
    mesh = trimesh.load(mesh_path, force='mesh')

    # Build a dataset instance to iterate cluster slices and recover face indices
    from mesh_dataset_j_4 import MeshDataset as VisDataset
    vis_ds = VisDataset(
        mesh_dir=mesh_dir,
        clusters_per_batch=clusters_per_batch,
        faces_per_cluster=faces_per_cluster,
        PE=use_pe,
        augmentation=None,
        transform=None,
        flexible_num_clusters=(n_clusters_fixed is None),
        n_clusters=n_clusters_fixed if n_clusters_fixed is not None else None,
    )

    # collect features per original face index
    num_faces = mesh.faces.shape[0]
    feature_dim_accum = None
    face_features = [None] * num_faces  # list of np arrays

    for idx in range(len(vis_ds)):
        clusters, cluster_face_ids = vis_ds[idx]
        # pack to tensor like training and build a full-visible mask
        max_seq = max(cl.size(0) for cl in clusters)
        feat_dim = clusters[0].size(1)
        if feature_dim_accum is None:
            feature_dim_accum = feat_dim
        # B=1, P=len(clusters), S=max_seq, F=feat_dim
        padded_data = [
            F.pad(cl, (0, 0, 0, max_seq - cl.size(0))) for cl in clusters
        ]
        batch_tensor = torch.stack(padded_data).unsqueeze(0).to(device)
        # masks: all true for real tokens; padded are false
        enc_masks = []
        for cl in clusters:
            mask = torch.zeros(max_seq, dtype=torch.bool)
            mask[:cl.size(0)] = True
            enc_masks.append(mask)
        enc_masks += [torch.zeros(max_seq, dtype=torch.bool)] * (len(padded_data) - len(clusters))
        encoder_mask = torch.stack(enc_masks).unsqueeze(0).to(device)

        # run encoder
        enc_out = target_encoder(batch_tensor, encoder_mask)  # [1,P,S,D]
        enc_np = enc_out.squeeze(0).cpu().numpy()

        # scatter back per real face
        for p, (cl, ids) in enumerate(zip(clusters, cluster_face_ids)):
            real_len = cl.size(0)
            for j in range(real_len):
                fid = int(ids[j].item())
                face_features[fid] = enc_np[p, j, :]

    # filter out any None (in case of inconsistencies)
    face_features_np = np.array([f for f in face_features if f is not None])
    if face_features_np.shape[0] != num_faces:
        print(f"Warning: collected {face_features_np.shape[0]} features, expected {num_faces}.")
    rgba = _pca_to_rgb(face_features_np)

    # assign per-face colors
    if rgba.shape[0] != mesh.faces.shape[0]:
        raise ValueError(f"Face color count ({rgba.shape[0]}) != num faces ({mesh.faces.shape[0]})")

    else:
        face_rgba = rgba
    mesh.visual.face_colors = face_rgba

    out_dir = os.path.join(save_root)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(out_dir, f"{base}_pca_colored.ply")
    mesh.export(out_path)
    print(f"Saved PCA-colored mesh to {out_path}")

if optimize_target:
    # include target_encoder parameters
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        context_encoder,
        context_predictor,
        target_encoder,
        start_lr,
        learning_rate,
        warmup,
        Training_epochs,
        iterations_per_epoch
    )
else:
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        context_encoder,
        context_predictor,
        None,
        start_lr,
        learning_rate,
        warmup,
        Training_epochs,
        iterations_per_epoch
    )

# Initialize TensorBoard SummaryWriter with custom log directory
writer = SummaryWriter(log_dir=log_dir)


accumulation_steps = 32



# Initialize variables for checkpoint tracking
best_loss = float('inf')
checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pth')

# Load checkpoint if resuming
if resume and os.path.exists(checkpoint_path):
    start_epoch, best_loss = load_checkpoint(
        context_encoder,
        context_predictor,
        target_encoder,
        optimizer,
        scheduler,
        checkpoint_path,
        iterations_per_epoch
    )
    print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
else:
    start_epoch = 0

def get_momentum(step, max_steps, base_m=0.9, final_m=1.0):
    return final_m - (final_m - base_m) * (0.5 * (1 + math.cos(math.pi * step / max_steps)))

# Create loss instance
def masked_smooth_l1_loss(predictions, targets, target_masks):
    """
    Calculate smooth L1 loss only between masked tokens and their targets.
    
    Args:
        predictions: Predicted embeddings [B, N, D]
        targets: Target embeddings [B, N, D]
        target_masks: Boolean mask indicating which tokens to predict [B, N]
    
    Returns:
        Loss value averaged over masked tokens
    """
    # Expand mask to match embedding dimension
    target_masks = target_masks.unsqueeze(-1).expand_as(predictions)
    
    # Calculate smooth L1 loss
    loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    
    # Apply mask and calculate mean only over masked tokens
    num_zeros = target_masks.logical_not().sum().item()
    num_nonzeros = (target_masks == 1).sum().item()
    masked_loss = (loss * target_masks.float()).sum() / (target_masks.float().sum() + 1e-8)
 #   masked_loss = loss.sum() / (target_masks.float().sum() + 1e-8)
    return masked_loss

import torch
import torch.nn.functional as F

def masked_cosine_loss(predictions, targets, target_masks, eps: float = 1e-8):
    """
    Cosine-based loss only over masked tokens.

    Args:
        predictions (Tensor): [B, N, D] — predicted embeddings
        targets     (Tensor): [B, N, D] — target embeddings
        target_masks(Tensor): [B, N] boolean mask (True for positions to include)
        eps         (float):   numerical eps for cosine_similarity

    Returns:
        loss (Tensor): scalar — average (1 − cosine_similarity) over masked tokens
    """
    # Compute cosine similarity per token: shape [B, N]
    cos_sim = F.cosine_similarity(predictions, targets, dim=-1, eps=eps)
    
    # Mask out the un-masked positions and flatten
    mask = target_masks.bool()
    masked_sims = cos_sim[mask]              # shape [num_masked]
    
    # If nothing is masked, return zero (or you could raise)
    if masked_sims.numel() == 0:
        return torch.tensor(0., device=predictions.device)
    
    # Loss = 1 − mean cosine similarity
    loss = (1.0 - masked_sims).mean()
    return loss
    
    
def masked_vicreg_loss(pred, target, target_masks, sim_coeff=0.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    """
    VICReg-style loss (invariance + variance + covariance) computed only on masked tokens.

    Args:
        pred:           [B, N, D] predicted embeddings
        target:         [B, N, D] target embeddings
        target_masks:   [B, N] boolean mask (True = masked token)
    Returns:
        VICReg loss scalar over masked positions
    """
    # Flatten
    flat_pred = pred.view(-1, pred.size(-1))       # [B*N, D]
    flat_target = target.view(-1, target.size(-1)) # [B*N, D]
    flat_mask = target_masks.view(-1).bool()       # [B*N]

    # Apply mask
    masked_pred = flat_pred[flat_mask]             # [M, D]
    masked_target = flat_target[flat_mask]         # [M, D]

    if masked_pred.size(0) < 2:
        # Not enough tokens for variance/covariance, return 0
        return torch.tensor(0.0, device=pred.device)

    # === Invariance loss ===
    inv_loss = F.mse_loss(masked_pred, masked_target)

    # === Variance loss ===
    def std_loss(x):
       std = torch.sqrt(x.var(dim=0) + eps)
       print(std.min(), std.max())
       return torch.mean(F.relu(1.0 - std))
        # target_cv = 2.5
        # mean = x.mean(dim=0)
        # std = torch.sqrt(x.var(dim=0) + eps)
        # cv = std / (mean.abs() + eps)  # coefficient of variation
        # print('CV',F.relu(target_cv - cv))
        # loss = torch.mean(F.relu(target_cv - cv))
        # return loss




    
    #    var_loss = std_loss(masked_pred) + std_loss(masked_target)
    var_loss = std_loss(masked_pred) 
    # === Covariance loss ===
    def cov_loss(x):
        x = x - x.mean(dim=0)
        N, D = x.size()
        print('N, D',N, D)
        cov = (x.T @ x) / (N - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D

#    cov_loss_val = cov_loss(masked_pred) + cov_loss(masked_target)
    cov_loss_val = cov_loss(masked_pred) 
    print("parts:",sim_coeff * inv_loss, std_coeff*var_loss, cov_coeff*cov_loss_val)  
    return sim_coeff * inv_loss + std_coeff * var_loss + cov_coeff * cov_loss_val
avg_l1_losses = []
avg_cosine_losses = []
avg_vicreg_losses = []
log_steps = []  # global step index for x-axis

# Training Loop
for epoch in range(start_epoch, Training_epochs):
    context_encoder.train()
    context_predictor.train()
    running_loss = 0.0
    running_l1 = running_cosine = running_vicreg = 0.0
    total_batches = 0
    # epoch‐level accumulators
    epoch_loss_sum = 0.0
    epoch_batches = 0

    # Wrap the data_loader with tqdm
    with tqdm(total=iterations_per_epoch,
              desc=f'Epoch {epoch+1}/{Training_epochs}',
              unit='batch') as pbar:
              
        for i, (batch, context_masks, target_masks, added_token_masks) in enumerate(data_loader):
            scheduler.step()
            print("Number of 1s in target_masks:", target_masks.sum().item())

      #      print('batch, context_masks, target_masks, added_token_masks',batch.shape, context_masks.shape, target_masks.shape, added_token_masks.shape)
            wd_scheduler.step()
            batch = batch.to(device)
            flat_input = batch.view(-1, batch.size(-1))  # [B*N*F, D_in]
            std_input = flat_input.std(dim=0).mean().item()
            print("std_input", std_input)
            input_norm = flat_input.norm(dim=-1).mean().item()
            print("avg input norm", input_norm)


            context_masks = context_masks.to(device)
            target_masks = target_masks.to(device)      
            added_token_masks = added_token_masks.to(device) 
            
            # Encode context
            context_encoded = context_encoder(batch, context_masks)
            # Predict masked regions
            context_prediction = context_predictor(batch,
                                                   context_encoded,
                                                   context_masks,
                                                   target_masks)
            # Target encoding
            target_encoded = target_encoder(batch, added_token_masks)  
            target_encoded = torch.repeat_interleave(
                target_encoded,
                num_mask_blocks,
                dim=1
            )
            Mb, Mrept, Mc, Mf = target_masks.shape 
            target_masks = target_masks.permute(0, 2, 1, 3).reshape(Mb, Mrept * Mc, Mf)
            
            with torch.no_grad():
                # Flatten embeddings to shape [B*N, D]
                flat_context = context_encoded.view(-1, context_encoded.size(-1))
                flat_prediction = context_prediction.view(-1, context_prediction.size(-1))
                flat_target = target_encoded.view(-1, target_encoded.size(-1))
                # Compute std deviation for each embedding dimension
                std_context = flat_context.std(dim=0).mean().item()
                std_prediction = flat_prediction.std(dim=0).mean().item()
                std_target = flat_target.std(dim=0).mean().item()
                prediction_norm = flat_prediction.norm(dim=-1).mean()
                target_norm = flat_target.norm(dim=-1).mean()
                print('prediction_norm, target_norm', prediction_norm, target_norm)
                print('std_prediction, std_target', std_prediction, std_target)
                # Optional: Print warning if variance drops too low
                if std_context < 1e-3 or std_prediction < 1e-3 or std_target < 1e-3:
                    print(f"⚠️  Warning: Potential collapse detected (very low variance). Epoch {epoch}, Batch {i}")
                std_prediction, std_target = compute_masked_std(
                    context_encoded,
                    context_prediction,
                    target_encoded,
                    target_masks
                )
                print(f'std_prediction, std_target (masked only): {std_prediction:.6f}, {std_target:.6f}')
                if std_prediction < 1e-3 or std_target < 1e-3:
                    print(f"⚠️  Warning: very low variance in masked outputs. Epoch {epoch}, Batch {i}")

            # Calculate loss only on the masked regions using our custom loss function
            loss_l1     = masked_smooth_l1_loss(context_prediction,
                                                target_encoded,
                                                target_masks)
            loss_cosine = masked_cosine_loss(context_prediction,
                                             target_encoded,
                                             target_masks)
            loss_vicreg = masked_vicreg_loss(context_prediction,
                                             target_encoded,
                                             target_masks)
            loss =  alpha * loss_l1 + gamma * loss_vicreg
            print('loss_l1, loss_cosine, loss_vicreg, loss', loss_l1, loss_cosine, loss_vicreg, loss)
            # backward + step
            # 1) Zero‐out any old gradients
            optimizer.zero_grad()
            
            # 2) BACKWARD on loss_l1 only, but KEEP the graph so we can do vicreg next
        #    loss_l1.backward(retain_graph=True)
            
            # 3) Measure how big the L1‐only gradients are
        #    l1_grad_norm = 0.0
        #    for p in list(context_encoder.parameters()) + list(context_predictor.parameters()):
        #        if p.grad is not None:
        #            l1_grad_norm += p.grad.norm().item()
        #    print(f"L1‐only grad norm: {l1_grad_norm:.4f}")



            # 4) Zero grads again, then BACKWARD on loss_vicreg only (keep graph for final pass)
        #    optimizer.zero_grad()
        #    loss_vicregg = gamma * loss_vicreg
        #    loss_vicregg.backward(retain_graph=True)
            
        #    # 5) Measure the VICReg‐only gradient norm
        #    vicreg_grad_norm = 0.0
        #    for p in list(context_encoder.parameters()) + list(context_predictor.parameters()):
        #        if p.grad is not None:
        #            vicreg_grad_norm += p.grad.norm().item()
        #    print(f"VICReg‐only grad norm: {vicreg_grad_norm:.4f}")
            
            
            # # collect the parameters you really care about
            # params = list(context_encoder.parameters()) + list(context_predictor.parameters())
            
            # # compute per‑loss gradients
            # g1 = torch.autograd.grad(loss_l1,      params, retain_graph=True, allow_unused=True)
            # g2 = torch.autograd.grad(loss_vicreg,  params, retain_graph=True, allow_unused=True)
            
            # # flatten and measure cosine similarity
            # flat_g1 = torch.cat([g.view(-1) for g in g1 if g is not None])
            # flat_g2 = torch.cat([g.view(-1) for g in g2 if g is not None])
            # cos = F.cosine_similarity(flat_g1, flat_g2, dim=0)
            # print("Grad alignment:", cos.item())
            
            # # 1) Compute per‑loss, per‑param grads (you already have g1, g2)
            # params = list(context_encoder.parameters()) + list(context_predictor.parameters())
            # # How large is VICReg’s own gradient?
            # vicreg_grad_norm = torch.norm(flat_g2).item()
            # # How large is L1's?
            # l1_grad_norm = torch.norm(flat_g1).item()
            # # Alignment angle
            # angle = torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)) * 180 / math.pi
            # print(f"L1 grad norm: {l1_grad_norm:.4f}, VICReg grad norm: {vicreg_grad_norm:.4f}, Angle: {angle:.2f}°")

        #    g1 = torch.autograd.grad(loss_l1,     params, retain_graph=True, allow_unused=True)
        #    g2 = torch.autograd.grad(loss_vicreg, params, retain_graph=True, allow_unused=True)
            
        #    # 2) PCGrad projection
        #    projected_g1 = []
        #    for grad1, grad2 in zip(g1, g2):
        #        if grad1 is None or grad2 is None:
        #            projected_g1.append(grad1 if grad1 is not None else None)
        #            continue
        #        # flatten both
        #        v1 = grad1.view(-1)
        #        v2 = grad2.view(-1)
        #        # only project when they conflict
        #        if torch.dot(v1, v2) < 0:
        #            proj_coeff = torch.dot(v1, v2) / (v2.norm()**2 + 1e-12)
        #            grad1 = grad1 - proj_coeff * grad2
        #        projected_g1.append(grad1)
            
            # 3) Write the combined gradient back into .grad
        #    for p, grad1, grad2 in zip(params, projected_g1, g2):
        #        if grad1 is None and grad2 is None:
        #            continue
        #        # combine with your weights α, γ
        #        p.grad = alpha * (grad1 if grad1 is not None else 0) \
        #               + gamma * (   grad2 if grad2 is not None else 0)
            
            # 4) Do the step
            loss.backward()
            optimizer.step()
   


            # If frozen, update via momentum; otherwise let optimizer handle it
            if not optimize_target:
                step = epoch * iterations_per_epoch + i
                momentum = get_momentum(step,
                                        Training_epochs * iterations_per_epoch)
                update_target_encoder(context_encoder,
                                      target_encoder,
                                      momentum)
            
            # Update running loss
            # Accumulate for averaging
            running_l1 += loss_l1.item()
            running_cosine += loss_cosine.item()
            running_vicreg += loss_vicreg.item()
            running_loss += loss.item()
            total_batches += 1
            epoch_loss_sum += loss.item()
            epoch_batches += 1
            # Update progress bar
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
            
            # Log every 10 batches
            if i % 10 == 9:
                avg_loss = running_loss / total_batches
                step_l = epoch * iterations_per_epoch + i
                avg_l1 = running_l1 / total_batches
                avg_cos = running_cosine / total_batches
                avg_vic = running_vicreg / total_batches
                avg_l1_losses.append(avg_l1)
                avg_cosine_losses.append(avg_cos)
                avg_vicreg_losses.append(avg_vic)
                log_steps.append(step_l)

                # Reset accumulators
                running_l1 = running_cosine = running_vicreg = 0.0
                total_batches = 0
                writer.add_scalar('Loss/train', avg_loss, epoch * len(data_loader) + i)
                running_loss = 0.0
                total_batches = 0
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Average Loss: {avg_loss:.4f}")
        # Log final loss for the epoch
        avg_epoch_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else 0
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = os.path.join(checkpoint_dir, 'best')
            save_checkpoint(
                context_encoder,
                context_predictor,
                target_encoder,
                optimizer,
                scheduler,
                epoch + 1,
                best_loss,
                checkpoint_dir,
                'best_model.pth',
                best_mode=True,
                best_dir=best_dir
            )
            print(f"New best loss: {best_loss:.4f}. Checkpoint saved.")



        # 1) always save “last” checkpoint of this epoch
        last_ckpt_name = 'last_model.pth'
        save_checkpoint(
            context_encoder,
            context_predictor,
            target_encoder,
            optimizer,
            scheduler,
            epoch + 1,               # current epoch number
            avg_epoch_loss,          # loss for this epoch (or pass best_loss, up to you)
            checkpoint_dir,
            last_ckpt_name
        )
        print(f"Last‐epoch checkpoint saved: {last_ckpt_name}")
        
                # Save checkpoint every 50 epochs regardless of loss, with epoch in filename
        if (epoch + 1) % 50 == 0:
            epoch_ckpt_name = f'epoch_{epoch + 1}.pth'
            save_checkpoint(
                context_encoder,
                context_predictor,
                target_encoder,
                optimizer,
                scheduler,
                epoch + 1,
                avg_epoch_loss,
                checkpoint_dir,
                epoch_ckpt_name
            )
            print(f"Epoch {epoch + 1} checkpoint saved: {epoch_ckpt_name}")
        
        
        # After each epoch, plot each loss in its own figure
        # 1) L1 Loss
        plt.figure()
        plt.plot(log_steps, avg_l1_losses, label='L1 Loss (avg10)')
        plt.xlabel('Iteration')
        plt.ylabel('L1 Loss')
        plt.title('L1 Loss (latest)')
        plt.legend()
        fig_path = os.path.join(checkpoint_dir, 'l1_loss.png')  # Overwrite each epoch
        plt.savefig(fig_path)
        plt.close()

        # 2) Cosine Loss
        plt.figure()
        plt.plot(log_steps, avg_cosine_losses, label='Cosine Loss (avg10)')
        plt.xlabel('Iteration')
        plt.ylabel('Cosine Loss')
        plt.title('Cosine Loss (latest)')
        plt.legend()
        fig_path = os.path.join(checkpoint_dir, 'cosine_loss.png')  # Overwrite each epoch
        plt.savefig(fig_path)
        plt.close()

        # 3) VICReg Loss
        plt.figure()
        plt.plot(log_steps, avg_vicreg_losses, label='VICReg Loss (avg10)')
        plt.xlabel('Iteration')
        plt.ylabel('VICReg Loss')
        plt.title('VICReg Loss (latest)')
        plt.legend()
        fig_path = os.path.join(checkpoint_dir, 'vicreg_loss.png')  # Overwrite each epoch
        plt.savefig(fig_path)
        plt.close()

        # 4) Total Loss
        if len(log_steps) == len(avg_l1_losses) == len(avg_vicreg_losses):
            avg_total_losses = [alpha * l1 + gamma * vic for l1, vic in zip(avg_l1_losses, avg_vicreg_losses)]
            plt.figure()
            plt.plot(log_steps, avg_total_losses, label='Total Loss (avg10)')
            plt.xlabel('Iteration')
            plt.ylabel('Total Loss')
            plt.title('Total Loss (latest)')
            plt.legend()
            fig_path = os.path.join(checkpoint_dir, 'total_loss.png')  # Overwrite each epoch
            plt.savefig(fig_path)
            plt.close()


        # # Every 10 epochs: visualize PCA coloring on a random mesh and save
        # if (epoch + 1) % 1 == 0:
        #     save_root = os.path.join(checkpoint_dir, 'pca_visuals', f'epoch_{epoch + 1}')
        #     prev_mode = target_encoder.training
        #     target_encoder.eval()
        #     try:
        #         visualize_random_mesh_pca(
        #             target_encoder=target_encoder,
        #             mesh_dir=mesh_dir,
        #             save_root=save_root,
        #             device=device,
        #             faces_per_cluster=faces_per_cluster,
        #             clusters_per_batch=clusters_per_batch,
        #             use_pe=PE,
        #             n_clusters_fixed=n_clusters,
        #         )
        #     finally:
        #         if prev_mode:
        #             target_encoder.train()

print('Training complete')
# Close the TensorBoard writer
writer.close()
