# Downstream Semantic Segmentation with IJEPA

This script implements downstream semantic segmentation using a pre-trained IJEPA model. It adds a classification head on top of the pre-trained encoder to perform per-face semantic segmentation tasks.

## Overview

The downstream training script (`train_downstream.py`) does the following:

1. **Loads pre-trained IJEPA encoder**: Uses the encoder from your pre-trained IJEPA model
2. **Freezes encoder parameters**: The pre-trained encoder weights are frozen during downstream training
3. **Adds per-face classification head**: A multi-layer classifier is added on top of the encoder
4. **Per-face prediction**: Predicts semantic labels for each individual face in the mesh
5. **Semantic segmentation**: Outputs predictions for every face, not just global mesh classification

## Key Components

### DownstreamClassifier
- Takes the pre-trained encoder as input
- Freezes encoder parameters to preserve learned representations
- Adds a per-face classification head with multiple layers:
  - Linear(embedding_dim → embedding_dim//2)
  - LayerNorm + GELU + Dropout
  - Linear(embedding_dim//2 → embedding_dim//4)
  - LayerNorm + GELU + Dropout
  - Linear(embedding_dim//4 → num_classes)
- Outputs predictions for each face: `[B, N, F, num_classes]`

### MeshDataset
- Uses the existing MeshDataset with labels
- Supports both .txt and .npy label files
- Provides per-face labels for semantic segmentation

### Per-Face Prediction Strategy
- No global pooling - each face gets its own prediction
- Handles padded faces by ignoring them in loss calculation (using -100 as ignore index)
- Maintains the hierarchical structure: batch → clusters → faces → classes

## Usage

### 1. Prepare your data
```
train_meshes/
├── mesh1.obj
├── mesh2.obj
└── ...

train_labels/
├── mesh1.txt  # per-face labels
├── mesh2.txt
└── ...

val_meshes/
├── mesh_val1.obj
├── mesh_val2.obj
└── ...

val_labels/
├── mesh_val1.txt
├── mesh_val2.txt
└── ...
```

### 2. Update configuration
Edit `config_downstream.yaml`:
- Set correct paths for your data
- Adjust `n_classes` to match your semantic segmentation task
- Modify training parameters as needed

### 3. Run training
```bash
python train_downstream.py config_downstream.yaml --pretrained_path path/to/ijepa_checkpoint.pth
```

## Configuration

Key parameters in `config_downstream.yaml`:

- `n_classes`: Number of semantic classes for your segmentation task
- `learning_rate`: Learning rate for the classifier (encoder is frozen)
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

## Expected Label Format

### Text files (.txt)
Per-face labels for each mesh:
```
0 1 2 0 1 2 0 1 2 ...  # One label per face in the mesh
```

### NumPy files (.npy)
Array of per-face labels:
```python
import numpy as np
# For each mesh, create an array of face labels
face_labels = np.array([0, 1, 2, 0, 1, 2, ...])  # One label per face
np.save('mesh1_labels.npy', face_labels)
```

## Training Process

1. **Pre-trained encoder**: Loads the encoder from your IJEPA checkpoint
2. **Frozen encoder**: Encoder parameters are frozen during training
3. **Per-face classifier training**: Only the classification head is trained
4. **Semantic segmentation**: Predicts class for each face individually
5. **Metrics**: Tracks accuracy and F1 score for face-level predictions
6. **Checkpointing**: Saves best model based on validation F1 score

## Output

- **Logs**: TensorBoard logs for monitoring training progress
- **Checkpoints**: Best model and regular checkpoints every 5 epochs
- **Metrics**: Training and validation loss, accuracy, and F1 score (per-face)
- **Predictions**: Per-face semantic labels for each mesh

## Key Differences from Global Classification

1. **No pooling**: Each face gets its own prediction instead of global mesh classification
2. **Per-face loss**: Loss is calculated for each face individually
3. **Padded face handling**: Uses -100 as ignore index to exclude padded faces from loss
4. **Semantic segmentation**: Output shape is `[B, N, F, num_classes]` instead of `[B, num_classes]`

## Tips

1. **Learning rate**: Use a lower learning rate than IJEPA training since you're fine-tuning
2. **Batch size**: May need to reduce batch size depending on your GPU memory
3. **Label consistency**: Ensure face labels are consistent across your dataset
4. **Validation**: Monitor validation metrics to avoid overfitting
5. **Memory usage**: Per-face prediction uses more memory than global classification

## Example Output
```
Epoch 1: Train Loss: 2.3026, Train Acc: 0.1000, Train F1: 0.1000
          Val Loss: 2.3012, Val Acc: 0.1000, Val F1: 0.1000
New best F1: 0.1000. Checkpoint saved.