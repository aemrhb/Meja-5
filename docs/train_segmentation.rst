Train Segmentation
==================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

This document describes the end-to-end process for training a semantic segmentation model, including environment setup, dataset preparation, configuration, training, evaluation, inference, logging, and troubleshooting. It follows Sphinx-friendly reStructuredText formatting so it can be included directly in the project documentation.

Key features
~~~~~~~~~~~~

- Multi-class semantic segmentation training
- Configurable model backbone, optimizer, scheduler, and augmentations
- Mixed precision (AMP) support
- Distributed and multi-GPU training (if available)
- Checkpointing, early stopping, and resume
- Built-in evaluation metrics (mIoU, pixel accuracy, Dice/F1)

Prerequisites
-------------

- Python 3.9+
- GPU with CUDA support (optional but recommended)
- PyTorch or your chosen deep learning framework installed
- Sufficient disk space for datasets and checkpoints

Install
-------

Install project in editable mode and dependencies:

.. code-block:: bash

   cd /home/nhgnheid/meja_5
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   # If there is a requirements file
   # pip install -r requirements.txt

End-to-End Training Procedure
-----------------------------

This section summarizes how training runs in this codebase from start to finish.

1. Configuration Load
   - A YAML config is passed to the training script. Paths (train/val meshes, labels, textures), hyperparameters (epochs, batch size), optimization (optimizer, weight decay), and model settings (encoder dims/heads/blocks, texture options) are read.

2. Dataset and Dataloaders
   - Geometry-only: `MeshDataset` + `custom_collate_fn` yield `(batch, labels, masks)` where:
     - `batch`: `[B, N, F, Dg]`, `labels`: `[B, N, F]`, `masks`: `[B, N, F]` (1=valid)
   - Texture+Geometry: `MeshTextureDataset` + `texture_custom_collate_fn` yield `(geometry_features, labels, texture_sequences, masks, texture_masks)`.
   - Validation dataloaders are built similarly with val paths.

3. Model Construction
   - Geometry-only: `nomeformer` from `model_G_2.py` produces embeddings, wrapped by `DownstreamClassifier` to output per-face logits `[B, N, F, C]`.
   - Integrated: `IntegratedTextureGeometryModel` fuses geometry and texture; `IntegratedDownstreamClassifier` outputs logits.
   - Optional: load pretrained encoder weights if provided; otherwise apply Xavier/LayerNorm init. Optionally freeze encoder layers or unfreeze all.

4. Optimizer, LR Schedule, and EMA
   - Optimizer: AdamW with decay/no-decay param groups (bias/norm excluded from decay).
   - Optional layer-wise LR decay: deeper blocks get higher LR; head multiplier can boost classifier.
   - Scheduler: CosineAnnealingLR over half the total epochs (`eta_min=1e-6`).
   - EMA: maintain exponential moving average of model params for more stable evaluation.

5. Mixed Precision and Training Loop
   - AMP enabled by default: forward, loss, and backward are performed with `autocast` and `GradScaler` for speed/memory benefits.
   - Per batch:
     - Forward pass to get `logits` `[B, N, F, C]`.
     - Loss via `MaskedCrossEntropyLoss(logits, labels, masks)` applying the face-validity mask and optional `ignore_index`.
     - Backward pass and optimizer step (scaled if AMP). EMA is updated after step.
     - Metrics are updated only over valid faces: per-class F1, Accuracy, and IoU are tracked.

6. Validation and Checkpointing
   - At epoch end: compute validation loss, mean F1, mean Accuracy, and mIoU. If EMA is enabled, it is applied during evaluation and then restored.
   - If validation mean F1 improves, save a checkpoint containing model, optimizer, scheduler, best score, and EMA state.
   - Scheduler steps once per epoch.

7. Resuming Training
   - If `training.resume` is true and a checkpoint is found, model/scheduler/optimizer/best score/EMA are restored, and training continues from the stored epoch.

8. Logging
   - TensorBoard scalars: `Loss/train`, `Loss/val`, `F1/train`, `F1/val`, `Accuracy/train`, `Accuracy/val`.
   - Console prints include shapes, per-batch summaries, and best-checkpoint updates.

Functions Used in the Procedure
-------------------------------

This section explains the primary functions/classes invoked during each phase of training.

1) Configuration Load
~~~~~~~~~~~~~~~~~~~~~

- ``yaml.safe_load``: Parse the YAML file into a Python dictionary for paths, hyperparameters, and model options.

2) Dataset and Dataloaders
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``MeshDataset`` (``mesh_dataset_2.py``): Loads geometry-only samples; returns per-face features and labels.
- ``custom_collate_fn`` (``mesh_dataset_2.py``): Pads variable-size clusters into tensors and builds binary face masks.
- ``MeshTextureDataset`` (``mesh_texture_dataset.py``): Loads geometry plus per-face texture sequences.
- ``texture_custom_collate_fn`` (``mesh_texture_dataset.py``): Collates geometry and texture streams and masks.

3) Model Construction
~~~~~~~~~~~~~~~~~~~~~

- ``nomeformer`` (``model_G_2.py``): Transformer-style encoder for mesh geometry; outputs per-face embeddings.
- ``IntegratedTextureGeometryModel``: Encodes geometry and texture; fuses into a joint representation.
- ``DownstreamClassifier`` (``tools/downst.py``) / ``IntegratedDownstreamClassifier``: Map embeddings to per-face logits.
- ``_init_enc_weights`` (training script helper): Initializes Linear/LayerNorm layers when no pretrained is used.

4) Optimizer, LR Schedule, and EMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``_split_decay_param_groups`` (training script helper): Splits params into decay/no-decay groups (bias/norm excluded).
- ``torch.optim.AdamW``: Optimizer with decoupled weight decay.
- ``torch.optim.lr_scheduler.CosineAnnealingLR``: Cosine LR schedule over training.
- ``EMA`` (training script): Tracks exponential moving averages of parameters for stable evaluation.

5) Mixed Precision and Training Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``torch.cuda.amp.autocast`` and ``GradScaler``: Mixed-precision forward/backward to save memory and speed up.
- Forward pass (model call): Produces ``logits`` of shape ``[B, N, F, C]``.
- ``MaskedCrossEntropyLoss`` (``loss.py``): Computes loss over valid faces; supports ``ignore_index``.
- Metrics: ``F1Score`` and ``Accuracy`` (TorchMetrics), ``IoU`` over ``ConfusionMatrix`` (Ignite) using only valid faces.

6) Validation and Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``evaluate`` (training script): Runs validation (optionally with EMA weights) and returns loss, F1, Accuracy, mIoU, and confusion matrix.
- ``save_checkpoint`` (``tools/check_point.py``): Saves model, optimizer, scheduler, best score, and optional EMA state when validation improves.

7) Resuming Training
~~~~~~~~~~~~~~~~~~~~

- ``torch.load``: Loads a checkpoint file from disk.
- ``load_state_dict``: Restores model/optimizer/scheduler/EMA states for continued training.

8) Logging
~~~~~~~~~~

- ``SummaryWriter``: Writes train/val scalars (loss, F1, accuracy) to TensorBoard.
- ``tqdm`` and ``print``: Show progress bars and textual summaries during training and validation.

Dataset Preparation
-------------------

Expected directory structure (example):

.. code-block:: text

   data/
     └── segmentation_dataset/
         ├── images/
         │   ├── train/
         │   ├── val/
         │   └── test/
         └── masks/
             ├── train/
             ├── val/
             └── test/

- images: RGB or grayscale input images
- masks: per-pixel labels with class indices (0..N-1). Ensure mask encoding matches the configuration (e.g., background = 0).

Optionally provide a class map file:

.. code-block:: text

   classes.txt
   0 background
   1 road
   2 building
   3 vegetation

Configuration
-------------

All training options should be specified via a configuration file (YAML/TOML/JSON). A typical configuration may include:

.. code-block:: yaml

   experiment_name: seg_baseline

   dataset:
     root: data/segmentation_dataset
     train_split: train
     val_split: val
     num_classes: 4
     image_size: [512, 512]
     normalization:
       mean: [0.485, 0.456, 0.406]
       std:  [0.229, 0.224, 0.225]

   augmentations:
     horizontal_flip: true
     color_jitter: {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1}
     random_crop: {size: [512, 512], padding: 16}

   model:
     name: unet
     backbone: resnet34
     pretrained: true
     dropout: 0.1

   optimizer:
     name: adamw
     lr: 0.0003
     weight_decay: 0.01

   scheduler:
     name: cosine
     warmup_epochs: 5
     max_epochs: 100

   training:
     epochs: 100
     batch_size: 8
     num_workers: 8
     amp: true
     seed: 42
     gradient_clip_val: 1.0
     accumulate_grad_batches: 1
     early_stopping: {patience: 10, metric: val/mIoU, mode: max}

   checkpointing:
     dir: runs/checkpoints/seg_baseline
     save_top_k: 3
     monitor: val/mIoU
     mode: max

   logging:
     dir: runs/logs/seg_baseline
     log_every_n_steps: 50

Command Line Interface
----------------------

Assuming the project provides a CLI entry point (replace with your actual command/module):

.. code-block:: bash

   # Single-GPU or CPU
   python -m training.train_segmentation --config configs/seg_baseline.yaml

   # Multi-GPU (Distributed Data Parallel) with torchrun
   torchrun --nproc_per_node=4 -m training.train_segmentation --config configs/seg_baseline.yaml

   # Overriding config values from the command line (if supported)
   python -m training.train_segmentation --config configs/seg_baseline.yaml training.epochs=200 optimizer.lr=0.0001

Training Loop Summary
---------------------

Typical training steps each epoch:

1. Load batches of images and masks
2. Forward pass through the model
3. Compute loss (e.g., CrossEntropy, Dice, or combo)
4. Backpropagate and update parameters
5. Update learning rate scheduler
6. Evaluate on validation set periodically
7. Save best checkpoints based on monitored metric

Loss Functions
--------------

Common choices for segmentation:

- Cross Entropy Loss (multi-class)
- Dice Loss / Soft Dice
- Focal Loss
- Combo Losses (e.g., Cross Entropy + Dice)

Example PyTorch-style composite loss (pseudo-code):

.. code-block:: python

   def compute_loss(logits, targets):
       ce = cross_entropy_loss(logits, targets)
       dice = dice_loss_from_logits(logits, targets, num_classes)
       return 0.5 * ce + 0.5 * dice

Metrics
-------

- Pixel Accuracy
- Mean Intersection-over-Union (mIoU)
- Frequency Weighted IoU (FWIoU)
- Dice Coefficient (per-class and mean)

Evaluation
----------

During validation, compute metrics over the entire validation set. Recommended protocol:

- Resize or center-crop consistently with training preprocessing
- Disable augmentations for validation
- Use sliding-window or tiling for very large images (optional)
- Accumulate confusion matrices per class to compute mIoU

Inference
---------

Run inference on a directory of images:

.. code-block:: bash

   python -m training.infer_segmentation \
     --checkpoint runs/checkpoints/seg_baseline/best.ckpt \
     --images data/segmentation_dataset/images/test \
     --output runs/predictions/seg_baseline \
     --image-size 512 512

Optional post-processing:

- Argmax over logits to obtain class indices
- Morphological operations (opening/closing) to reduce noise
- CRF refinement (if required)

Checkpoints and Resume
----------------------

- Best checkpoints are saved based on the monitored validation metric
- To resume training:

.. code-block:: bash

   python -m training.train_segmentation --config configs/seg_baseline.yaml --resume runs/checkpoints/seg_baseline/last.ckpt

Reproducibility
---------------

- Set a global seed for Python, NumPy, and your DL framework
- Log package versions and Git commit hash
- Fix CuDNN and deterministic flags when needed (may reduce speed)

.. code-block:: python

   def set_seed(seed: int):
       import os, random
       import numpy as np
       import torch
       random.seed(seed)
       np.random.seed(seed)
       os.environ["PYTHONHASHSEED"] = str(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

Performance Tips
----------------

- Use AMP (mixed precision) to accelerate on GPUs and reduce memory
- Adjust batch size to maximize GPU utilization without OOM
- Profile data loading; increase ``num_workers`` if input pipeline is a bottleneck
- Cache datasets or use memory-mapped formats for large datasets
- Prefer gradient accumulation over reducing image size when constrained by memory

Project Structure (Example)
---------------------------

.. code-block:: text

   meja_5/
     ├── configs/
     │   └── seg_baseline.yaml
     ├── training/
     │   ├── train_segmentation.py
     │   ├── data.py
     │   ├── models/
     │   └── utils.py
     ├── runs/
     │   ├── checkpoints/
     │   └── logs/
     └── docs/
         └── train_segmentation.rst

Troubleshooting
---------------

- Training is slow
  - Enable AMP, increase ``num_workers``, and verify GPU utilization
- Out-of-memory (OOM)
  - Reduce batch size, use gradient accumulation, ensure no large tensors are kept
- Validation metric not improving
  - Try different learning rates, schedulers, or augmentations; verify label encoding
- Misaligned masks
  - Ensure images and masks are paired consistently and transforms are applied identically

FAQ
---

- How do I add a new backbone?
  - Implement the backbone and expose a factory function, then reference it in the config.
- How do I add a new dataset?
  - Create a dataset class with ``__len__`` and ``__getitem__`` returning ``image, mask``; update the config.

Sphinx Integration
------------------

Add this page to your Sphinx ``toctree`` (e.g., ``index.rst``):

.. code-block:: rst

   .. toctree::
      :maxdepth: 2

      train_segmentation

API Reference (Optional)
------------------------

If your project exposes importable modules for training, you can include auto-generated API docs. Replace module paths with your actual package names.

.. code-block:: rst

   .. automodule:: training.train_segmentation
      :members:
      :undoc-members:
      :show-inheritance:

   .. automodule:: training.data
      :members:
      :undoc-members:
      :show-inheritance:

   .. automodule:: training.utils
      :members:
      :undoc-members:
      :show-inheritance:
