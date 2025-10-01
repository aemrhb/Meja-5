# Meja 5 — Segmentation Training and Downstream Pipeline

This repository contains code and utilities to train and evaluate a downstream semantic segmentation model on mesh-based geometry (and optional texture) inputs.

- Purpose
  - Provide an end-to-end pipeline to learn per-face semantic labels on 3D meshes using a transformer-style geometry encoder (`nomeformer`) with optional texture fusion. It covers dataset loading and batching for irregular mesh data, configurable model setup, mixed-precision training, optional EMA stabilization, comprehensive evaluation (F1, Accuracy, IoU), and robust checkpointing/resume. The code is designed for fine-tuning a pretrained encoder on downstream segmentation tasks and supports freezing/unfreezing strategies and layer-wise learning-rate decay.

- Main training script: `train_downstream_2.py`
- Geometry encoder: `model_G_2.py` (exports `nomeformer`)
- Datasets: `mesh_dataset_2.py`, `mesh_texture_dataset.py`
- Downstream heads and tools: `tools/`
- Sphinx docs: `docs/train_segmentation.rst`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # or: pip install -r requirements.txt
```

Train with a YAML config:

```bash
python train_downstream_2.py path/to/config.yaml
```

TensorBoard logs: set `paths.log_dir` in your config and run `tensorboard --logdir <log_dir>`.

## Configuration (YAML)

Key fields consumed by `train_downstream_2.py`:

```yaml
paths:
  mesh_dir: /path/to/train/meshes
  label_dir: /path/to/train/labels
  json_dir: /path/to/train/json
  log_dir: runs/logs/exp1
  checkpoint_dir: runs/checkpoints/exp1
  checkpoint_pertrain: runs/pretrain/encoder.ckpt
  val_mesh_dir: /path/to/val/meshes
  val_label_dir: /path/to/val/labels
  val_json_dir: /path/to/val/json
  # Optional for texture
  texture_dir: /path/to/train/textures
  val_texture_dir: /path/to/val/textures

training:
  epochs: 100
  batch_size: 8
  clusters_per_batch: 8
  accumulation_steps: 1
  resume: true
  use_pretrained: true
  num_unfrozen_encoder_layers: 0   # or 'all'
  weight_decay: 0.01
  ema_decay: 0.9999
  use_ema: true
  ignore_index: null               # set class index to ignore for metrics/loss
  layer_wise_lr_decay: [10, 2]     # optional [head_multiplier, decay_factor]

model:
  feature_dim: 64
  embedding_dim: 256
  num_heads: 8
  num_attention_blocks: 8
  n_classes: 10
  n_clusters: 64
  faces_per_cluster: 128
  use_pe: false
  gradinat_ac: false
  dropout: 0.1
  include_normals: true
  additional_geometrical_features: false
  # Texture
  use_texture: false
  texture_embed_dim: 64
  fusion_method: gated
  max_texture_pixels: 128
  texture_patch_size: 16

augmentation:
  rotation_range: [0, 0, 180]
  scale_range: [0.9, 1.1]
  noise_std: 0.0
  flip_probability: 0.0
  mode: none

optimizer:
  type: adamw
  learning_rate: 0.001
```

## Script Overview — `train_downstream_2.py`

`train_downstream_2.py` orchestrates: configuration, data loading, model creation (geometry-only or integrated texture+geometry), optimizer and LR scheduling, mixed-precision training, EMA tracking, evaluation, and checkpointing.

### CLI

```bash
python train_downstream_2.py CONFIG_PATH.yaml
```

- Positional argument: `config_path` — path to the YAML configuration.

### Major Components and Functions

- Class EMA
  - Purpose: Maintain an exponential moving average (EMA) of model parameters for more stable evaluation and improved generalization.
  - Methods:
    - `register()`: Register parameters to track.
    - `update()`: Update shadow parameters after each optimizer step.
    - `apply_shadow()`: Swap model params with EMA params for evaluation.
    - `restore()`: Restore the original (non-EMA) model parameters.
    - `state_dict()` / `load_state_dict(...)`: Serialize/restore EMA state for checkpoints.

- Function evaluate(model, val_data_loader, ema=None)
  - Runs validation. If `ema` is provided, applies EMA weights for evaluation.
  - Computes: loss (via `MaskedCrossEntropyLoss`), per-class F1, mean F1, per-class Accuracy, mean Accuracy, and mIoU (via `ignite.metrics.IoU`).
  - Handles `ignore_index` so metrics can exclude an unlabeled class.
  - Returns: `(val_loss, val_mean_f1_score, val_mean_accuracy, val_miou, val_f1_scores, val_confusion_matrix)`.

- Function monitor_gpu_memory()
  - Prints and returns allocated/reserved/total memory (in GB) when CUDA is available.

- Function _init_enc_weights(m)
  - Helper used when skipping pretrained weights to initialize `nn.Linear` and `nn.LayerNorm` layers.

- Function _split_decay_param_groups(all_named_parameters, candidate_params=None)
  - Splits parameters into weight-decayed and non-decayed groups based on names (biases and norm weights excluded from decay).
  - Used to build optimizer parameter groups, supports layer-wise LR decay.

- Training Loop
  - Mixed precision with `torch.cuda.amp` (forward, loss compute, backward scaling via `GradScaler`).
  - Supports both geometry-only and integrated texture+geometry input pipelines.
  - Computes and logs metrics with TorchMetrics and Ignite; logs to TensorBoard.
  - Scheduler: CosineAnnealingLR (`T_max = epochs/2`, `eta_min = 1e-6`).
  - Checkpointing: saves on best validation mean F1; optionally stores EMA state.

### Data Loading

- Geometry only: `MeshDataset` with `custom_collate_fn`.
- Texture+Geometry: `MeshTextureDataset` with `texture_custom_collate_fn`.
- Both training and validation DataLoaders are built using config paths.

### Model Construction

- Geometry-only encoder: `nomeformer` from `model_G_2.py`.
- Texture+Geometry integrated model: `IntegratedTextureGeometryModel`.
- Downstream classifier heads:
  - Geometry-only: `DownstreamClassifier`
  - Integrated: `IntegratedDownstreamClassifier`
- Pretrained weights:
  - If `use_pretrained` and `paths.checkpoint_pertrain` exists, loads `checkpoint['target_encoder']` into the encoder.
  - Otherwise, initializes encoder via `_init_enc_weights`.
- Fine-tuning control:
  - `num_unfrozen_encoder_layers`: freezes most encoder layers; set `'all'` to unfreeze everything.

### Optimizer and Scheduling

- Base optimizer: `AdamW`.
- Parameter grouping:
  - Standard: splits into decay/no-decay groups by excluding biases and norm weights from weight decay.
  - Optional layer-wise LR decay: head gets `head_multiplier * lr`; encoder blocks use progressively smaller LRs moving toward input.
- Scheduler: `CosineAnnealingLR` with half-period over total epochs.

### Loss and Metrics

- Loss: `MaskedCrossEntropyLoss` (custom), respects `ignore_index`.
- Metrics:
  - Ignite: `IoU` over a `ConfusionMatrix`.
  - TorchMetrics: `F1Score` and `Accuracy` (per-class, mean aggregated).
- Mask handling:
  - Only valid faces (mask == 1) contribute to metrics.

### Checkpointing and Resume

- Auto-detects latest checkpoint in `paths.checkpoint_dir` matching `checkpoint_epoch{EPOCH}_f1_{SCORE}.pth`.
- On improvement of validation mean F1, saves new checkpoint; prints confusion matrix.
- If `training.resume` is true and checkpoint exists, restores model, optimizer (best-effort), scheduler, best F1, and EMA.

## Custom Modules (Authored in this Repo)

High-level descriptions of in-repo modules used by `train_downstream_2.py`:

- `model_G_2.py`
  - Exports `nomeformer`: a geometry-focused transformer-style encoder for mesh features. Configurable `feature_dim`, `embedding_dim`, `num_heads`, `num_attention_blocks`, `dropout`, `summary_mode`, and optional hierarchical processing and positional encodings.

- `mesh_dataset_2.py`
  - `MeshDataset`: Loads geometry-only samples; emits `(batch, labels, masks)` with optional normals and additional geometric features.
  - `custom_collate_fn`: Batches variable-size mesh clusters into padded tensors and masks.
  - `MeshAugmentation`: Configurable rotation/scale/noise/flip augmentations for meshes.

- `mesh_texture_dataset.py`
  - `MeshTextureDataset`: Extends geometry inputs with texture sequences and corresponding texture masks.
  - `texture_custom_collate_fn`: Collate function for combined geometry+texture batches.

- `tools/downst.py`
  - `DownstreamClassifier`: Classification head on top of the encoder output, producing per-face logits for `n_classes`.

- `integrated_texture_geometry_model.py`
  - `IntegratedTextureGeometryModel`: Encodes geometry and texture streams and fuses them (e.g., gated fusion).
  - `IntegratedDownstreamClassifier`: Downstream head compatible with the integrated encoder.

- `tools/check_point.py`
  - `save_checkpoint(model, optimizer, epoch, best_f1_score, checkpoint_dir, ema_state_dict=None, scheduler=None)`: Persists model, optimizer, EMA, scheduler, and metadata.
  - `load_checkpoint(...)`: Utility to restore saved artifacts (not directly used in the current script).

- `loss.py`
  - `MaskedCrossEntropyLoss(ignore_index=None)`: Wraps `nn.CrossEntropyLoss` and applies a binary mask over faces; supports ignoring a specific class label.

- `tools/helper.py`
  - `init_opt`: Helper to initialize optimizers/schedulers (imported but not used directly in `train_downstream_2.py`).

## Logging

- TensorBoard scalars:
  - `Loss/train`, `F1/train`, `Accuracy/train`
  - `Loss/val`, `F1/val`, `Accuracy/val`
- Console prints batch/epoch metrics and shapes for debugging.

## Sphinx Docs

A Sphinx-style guide for training segmentation is available at `docs/train_segmentation.rst`. To include in your Sphinx site:

```rst
.. toctree::
   :maxdepth: 2

   train_segmentation
```

## Tips

- Enable AMP for speed and memory savings (`use_mixed_precision=True` in the script). It is on by default.
- When seeing OOM, reduce `training.batch_size`, increase gradient accumulation, or reduce resolution/cluster sizes.
- Use `ignore_index` to exclude unlabeled background from metrics.

## License

Proprietary — internal research use only unless specified otherwise.


## Function I/O Reference

Below are key functions and classes used by `train_downstream_2.py` with their inputs/outputs and expected tensor shapes.

### Functions in `train_downstream_2.py`

- evaluate(model, val_data_loader, ema=None)
  - Inputs:
    - `model`: PyTorch module producing per-face logits `[B, N, F, C]`
    - `val_data_loader`: yields batches depending on dataset (see below)
    - `ema` (optional): EMA instance; if provided, applied during eval
  - Returns:
    - `val_loss` (float)
    - `val_mean_f1_score` (float)
    - `val_mean_accuracy` (float)
    - `val_miou` (float or tensor from Ignite IoU)
    - `val_f1_scores` (tensor `[C]` per-class F1)
    - `val_confusion_matrix` (Ignite ConfusionMatrix object)

- monitor_gpu_memory()
  - Inputs: none
  - Returns: `(allocated_gb: float, reserved_gb: float, total_gb: float)`; prints a summary when CUDA available

- _init_enc_weights(m)
  - Inputs: `m` (module); initializes `nn.Linear` and `nn.LayerNorm`
  - Returns: none (in-place)

- _split_decay_param_groups(all_named_parameters, candidate_params=None)
  - Inputs:
    - `all_named_parameters`: iterator of `(name, param)` from `model.named_parameters()`
    - `candidate_params` (optional): iterable of parameters to consider; if `None`, uses all
  - Returns: `(decay_params: list[Tensor], no_decay_params: list[Tensor])`

### Class in `train_downstream_2.py`

- EMA(model, decay=0.9999)
  - Constructor Inputs: `model` (Module), `decay` (float)
  - Methods:
    - `register()` -> None
    - `update()` -> None
    - `apply_shadow()` -> None
    - `restore()` -> None
    - `state_dict()` -> dict
    - `load_state_dict(state: dict)` -> None

### Models

- nomeformer (from `model_G_2.py`)
  - Forward (geometry-only):
    - Inputs: `batch` `[B, N, F, D_in]`, `masks` `[B, N, F]` (binary 0/1)
    - Output: `logits` `[B, N, F, C]`

- IntegratedTextureGeometryModel (from `integrated_texture_geometry_model.py`)
  - Forward (integrated):
    - Inputs:
      - `geometry_features` `[B, N, F, Dg]`
      - `texture_sequences` `[B, N, T, Dt]` (T = texture tokens/pixels)
      - `masks` `[B, N, F]` (binary 0/1)
      - `texture_masks` `[B, N, T]` (binary 0/1)
    - Output: fused embeddings used by `IntegratedDownstreamClassifier`

- DownstreamClassifier (from `tools/downst.py`)
  - Inputs: encoder outputs for geometry `[B, N, F, E]`
  - Output: `logits` `[B, N, F, C]`

- IntegratedDownstreamClassifier (from `integrated_texture_geometry_model.py`)
  - Inputs: fused encoder outputs `[B, N, F, E]`
  - Output: `logits` `[B, N, F, C]`

#### Model Inputs and Outputs (Details)

- Shapes use: `B` = batch, `N` = meshes per sample or time steps, `F` = faces per mesh, `C` = number of classes, `Dg` = geometry feature dim, `Dt` = per-texture-token dim, `E` = encoder embedding dim, `T` = number of texture tokens/pixels per face.
- Dtypes unless noted: floating tensors are `float32` (or `float16`/`bfloat16` under autocast); masks are `bool` or `uint8`; labels are `long`/`int64`.

- Geometry-only forward (e.g., `nomeformer`):
  - Input `batch`: `[B, N, F, Dg]` float. Geometry features per face.
  - Input `masks`: `[B, N, F]` bool/byte. `1` (true) = valid face, `0` = padded/invalid.
  - Output `logits`: `[B, N, F, C]` float. Class scores per face.

- Integrated texture+geometry forward (e.g., `IntegratedTextureGeometryModel` used in `train_downstream_2.py`):
  - Input `geometry_features`: `[B, N, F, Dg]` float. Per-face geometry descriptors.
  - Input `texture_sequences`: `[B, N, T, Dt]` float. Per-face sampled pixels/tokens (T may be capped by `max_texture_pixels`).
  - Input `masks`: `[B, N, F]` bool/byte. Face validity mask; `1` = valid.
  - Input `texture_masks`: `[B, N, T]` bool/byte. Texture token validity mask; `1` = valid.
  - Output (encoder/fused): `[B, N, F, E]` float, consumed by the downstream head.
  - Output `logits` (from `IntegratedDownstreamClassifier`): `[B, N, F, C]` float.

- Loss/targets:
  - `labels`: `[B, N, F]` long with values in `[0, C-1]` (or `ignore_index`).
  - `MaskedCrossEntropyLoss(logits, labels, masks)` applies face masks (and `ignore_index` when configured).

- Masking semantics:
  - Face-level operations always apply `masks`; invalid faces contribute zero to attention/reductions and are excluded from the loss.
  - Texture attention applies `texture_masks` to ignore padded tokens for each face.

- Mixed precision:
  - The training loop may wrap forward in autocast. Ensure inputs are on the same device and compatible dtypes. Logits are typically accumulated in fp32 for stability during loss/optimizer steps.

### Integrated Model Hyperparameters (Texture + Geometry)

Configure these under `model` in your YAML. The geometry encoder is `nomeformer`; the texture branch summarizes per-face pixels and fuses with geometry features before encoding.

- Geometry encoder
  - embedding_dim: hidden size for encoder blocks
  - num_heads: attention heads per block
  - num_attention_blocks: encoder depth
  - dropout: dropout in attention/FFN
  - Optional (in code):
    - summary token mode: Adds a learned summary token (like CLS) that attends to all per-face tokens so its output can be used as a compact representation for classification heads; reduces reliance on mean/max pooling and can improve stability on short sequences.
    - hierarchical/global blocks: Uses mostly local/hierarchical attention with periodic global-attention blocks to keep memory/compute manageable on large meshes while preserving long-range information flow.
    - positional encodings: Injects spatial/topological structure so attention respects mesh layout. Can be sinusoidal/learned absolute or relative (e.g., adjacency/geodesic-based); relative often works best for variable-size, irregular topology.

- Texture branch
  - use_texture: set `true` to enable texture+geometry
  - texture_embed_dim: hidden size for texture embeddings
  - max_texture_pixels: sampled pixels per face (controls speed/memory)
  - texture_patch_size: side length for face sampling (dataset-dependent)

- Fusion
  - fusion_method: `gated` | `concat` | `add`
    - gated: learns a gate between projected geometry and texture
    - concat: concatenates and projects back to `embedding_dim`
    - add: element-wise sum (dims must match)

Example (texture enabled):

```yaml
model:
  # Geometry
  feature_dim: 64
  embedding_dim: 256
  num_heads: 8
  num_attention_blocks: 8
  dropout: 0.1

  # Texture + fusion
  use_texture: true
  texture_embed_dim: 128
  max_texture_pixels: 256
  texture_patch_size: 16
  fusion_method: gated  # options: gated | concat | add
```

### Datasets and Collate

#### DataLoader usage (purpose and parameters)

- **Purpose**: Build batches from variable-size mesh samples using a custom collate that pads per-cluster/per-face dimensions so tensors align within a batch.
- **Example**:

```python
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=texture_custom_collate_fn)
```

- **Parameters**:
  - `dataset`: a `MeshTextureDataset` instance emitting tuples for geometry+texture.
  - `batch_size`: number of items per batch.
  - `shuffle`: shuffles items each epoch for better generalization.
  - `collate_fn`: function that pads/stacks nested lists into dense tensors; choose `custom_collate_fn` for geometry-only or `texture_custom_collate_fn` for geometry+texture.

## Dataset Caching

`MeshTextureDataset` uses a two-tier caching mechanism to accelerate repeated epochs and runs when augmentation is off.

- **Where cache lives**: For each `mesh_dir`, a directory `mesh_dir/.cluster_texture_cache` is created. Each mesh produces one pickle file (`.pkl`).
- **What is cached**: KMeans cluster assignment and ordering, face-level geometry features (angles, areas, optional normals), positional encoding coordinates, label/index maps, optional additional geometrical features (slope, height, roughness), and texture pixel sequences with texture masks.
- **When cache is used**: Only when `augmentation is None`. The loader first checks an in-memory map, then falls back to the on-disk pickle. If neither exists, it computes, writes to disk, and stores in memory.
- **When cache is bypassed**: If an `augmentation` callable is provided, samples are recomputed on the fly for every access and are not cached, to avoid mixing augmented/non-augmented payloads.
- **Cache key (filename)**: Encodes parameters that affect the payload. Changes to any of these will create a new cache file automatically:
  - `n_clusters`, `PE`, `include_normals`, `AGF`, `coords_use_cluster_center`, `pe_bbox_normalized`, `texture_patch_size`, `max_texture_pixels`.
- **Corruption handling**: If a pickle fails to load or isn't a dict, it is deleted and recomputed transparently.
- **Clearing cache**:
  - Remove the folder manually: `rm -rf <mesh_dir>/.cluster_texture_cache`
  - Or use the helper scripts in this repo such as `clean_mesh_cache.sh` / `clean_cache.sh` if they match your setup.

No configuration flag is required to enable caching; it is automatic given `augmentation=None`.

Non-technical overview: What is a custom dataset?

- Think of a dataset as a neatly organized collection of examples the model learns from. A "custom dataset" simply means we arranged our data in the exact way our model needs it. For 3D objects (meshes), each example includes:
  - The object's shape information (geometry), broken into small pieces called faces.
  - Optionally, color/appearance information (texture), like tiny image patches sampled on each face.
- We also include simple yes/no flags (masks) to say which pieces are real data versus padding, so the model ignores the filler.
- This custom packaging makes training reliable and efficient: every training batch has the same tidy structure, even when different objects have different numbers of faces or texture pixels.

- MeshDataset (from `mesh_dataset_2.py`)
  - __getitem__ Output: geometry sample
  - DataLoader batch (with `custom_collate_fn`) yields:
    - `batch` `[B, N, F, Dg]`
    - `labels` `[B, N, F]` (long, values in `[0, C-1]` or adjusted)
    - `masks` `[B, N, F]` (binary 0/1; 1 means valid face)

- MeshTextureDataset (from `mesh_texture_dataset.py`)
  - __getitem__ Output: geometry + texture sample
  - DataLoader batch (with `texture_custom_collate_fn`) yields:
    - `geometry_features` `[B, N, F, Dg]`
    - `labels` `[B, N, F]`
    - `texture_sequences` `[B, N, T, Dt]`
    - `masks` `[B, N, F]`
    - `texture_masks` `[B, N, T]`

  - What is this?
    - `MeshTextureDataset` extends the geometry-only dataset by also sampling per-face texture information (pixels/tokens) from the mesh's texture map(s). Each item carries both geometry descriptors and a sequence of texture tokens for every valid face, enabling the integrated texture+geometry model.
    - `geometry_features` are numeric descriptors per face (e.g., normals/curvatures or learned features) with dimension `Dg`.
    - `texture_sequences` contains up to `T` sampled texture tokens per face. Each token has `Dt` channels (e.g., RGB or learned embeddings). `T` is commonly bounded by `max_texture_pixels` for speed/memory.
    - `masks` marks which faces are valid (1) vs. padded (0) within each sample; downstream models ignore invalid faces.
    - `texture_masks` marks which texture tokens are valid (1) vs. padded (0) for each face; attention and pooling over texture tokens respect this mask.
    - `labels` is the per-face class target in `[0, C-1]` (or `ignore_index`), used by masked cross-entropy during training.

#### `MeshDataset` and `custom_collate_fn` (purpose and I/O)

- **Purpose**: Handle geometry-only samples with variable faces per cluster and variable clusters per sample by padding to the maximum sizes in the batch.
- **`custom_collate_fn` inputs**:
  - `batch`: list of `(nested_list_faces, nested_list_labels)` from `MeshDataset.__getitem__`.
- **Outputs**:
  - `padded_batches`: `[B, P, S, Dg]` float tensor (P=clusters per sample, S=faces per cluster).
  - `padded_labels`: `[B, P, S]` long tensor.
  - `masks`: `[B, P, S]` bool tensor where `1`=valid face.

#### `MeshTextureDataset` and `texture_custom_collate_fn` (purpose and I/O)

- **Purpose**: Extend the above to include per-face texture sequences, padding both face and pixel dimensions while preserving cluster structure.
- **`texture_custom_collate_fn` inputs**:
  - `batch`: list of `(nested_list_faces, nested_list_labels, nested_list_textures, nested_list_texture_masks)` from `MeshTextureDataset.__getitem__`.
- **Outputs**:
  - `geometry_features`: `[B, P, S, Dg]` float tensor.
  - `labels`: `[B, P, S]` long tensor.
  - `texture_sequences`: `[B, P, S, T, C]` float tensor (T=pixels per face, C=channels).
  - `masks`: `[B, P, S]` bool tensor for faces.
  - `texture_masks`: `[B, P, S, T]` bool tensor for texture pixels.

#### `MeshTextureDataset` constructor arguments

- **mesh_dir**: Directory of mesh files (`.obj`/`.ply`).
- **label_dir**: Directory of per-face label files (`.txt`, one index per face).
- **texture_dir**: Directory of texture payloads (`.pkl` per mesh; list of per-face RGB sequences). Fallback `*_pixels_test.pkl` is also supported.
- **n_clusters**: Number of KMeans clusters over vertices for partitioning faces; controls patching.
- **clusters_per_batch**: Number of clusters sampled per dataset item; defines how many clusters go into one sample for batching.
- **PE**: Whether to include positional encoding features (face coordinates) in geometry features.
- **json_dir (optional)**: Directory of extra per-mesh JSON features; loaded if present.
- **augmentation (optional)**: Callable applied to feature tensors post-cache. When set, caching is disabled and features are recomputed on the fly.
- **transform (optional)**: Callable applied to the nested lists just before return; use sparingly.
- **include_normals**: If true, appends mesh face normals to per-face features.
- **additional_geometrical_features**: If true, computes and appends slope, height (relative to ground plane/mesh), and roughness features.
- **texture_patch_size**: Intended size for texture patches; present for compatibility. Current pipeline keeps per-face pixel sequences and does not resize patches here.
- **max_texture_pixels**: Hard cap for pixels per face; longer sequences are truncated during texture load, collate pads shorter ones.
- **max_texture_pixels_per_face**: Currently unused. Prefer `max_texture_pixels`.
- **coords_use_cluster_center**: If true, positional coords are built relative to the selected cluster centroid; otherwise relative to the mesh center.
- **pe_bbox_normalized**: If true, uses bbox-normalized absolute vertex coords for PE instead of relative coords.

### Loss

- MaskedCrossEntropyLoss (from `loss.py`)
  - Inputs:
    - `logits` `[B, N, F, C]`
    - `labels` `[B, N, F]`
    - `masks` `[B, N, F]` (binary 0/1)
  - Behavior: applies mask to ignore invalid faces; respects `ignore_index` if provided
  - Output: scalar loss (tensor)

