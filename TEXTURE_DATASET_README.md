# MeshTextureDataset - Texture-Enhanced 3D Mesh Dataset

This dataset class extends the original mesh dataset to include texture information from .pkl files, enabling training of models that process both geometric and texture features.

## Overview

The `MeshTextureDataset` class loads:
- **Mesh data**: 3D geometry, faces, normals, and derived features
- **Texture data**: RGB pixel information for each face from .pkl files
- **Labels**: Face-level classification labels
- **Masks**: Validity masks for both geometry and texture data

## Texture Data Format

### Expected .pkl File Structure

Each .pkl file should contain a list where each element corresponds to a face in the mesh:

```python
[
    (num_pixels_1, [[R, G, B], [R, G, B], ...]),  # Face 0
    (num_pixels_2, [[R, G, B], [R, G, B], ...]),  # Face 1
    (num_pixels_3, [[R, G, B], [R, G, B], ...]),  # Face 2
    ...
]
```

### Example

```python
# Example texture data for a mesh with 3 faces
texture_data = [
    (18, [[44, 33, 29], [39, 28, 24], [45, 34, 30], [44, 33, 29], 
          [33, 24, 21], [36, 24, 22], [45, 33, 31], [48, 37, 33], 
          [39, 28, 24], [34, 22, 20], [41, 30, 26], [38, 27, 23], 
          [42, 31, 27], [40, 29, 25], [37, 26, 22], [43, 32, 28], 
          [46, 35, 31], [35, 24, 20]]),
    (12, [[50, 40, 35], [48, 38, 33], [52, 42, 37], [49, 39, 34], 
          [47, 37, 32], [51, 41, 36], [53, 43, 38], [46, 36, 31], 
          [54, 44, 39], [45, 35, 30], [55, 45, 40], [44, 34, 29]]),
    (0, [])  # Empty texture (no pixels)
]
```

## Usage

### Basic Usage

```python
from mesh_texture_dataset import MeshTextureDataset, texture_custom_collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = MeshTextureDataset(
    mesh_dir="/path/to/meshes",
    label_dir="/path/to/labels", 
    texture_dir="/path/to/textures",
    n_clusters=16,
    clusters_per_batch=4,
    PE=True,
    include_normals=True,
    texture_patch_size=16,
    max_texture_pixels=256
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=texture_custom_collate_fn
)

# Use in training loop
for batch in dataloader:
    padded_batches, padded_labels, padded_textures, masks, texture_masks = batch
    # Process your data...
```

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mesh_texture_dataset import MeshTextureDataset, texture_custom_collate_fn
from lightweight_texture_processor import LightweightTextureGeometryModel

# Create dataset
dataset = MeshTextureDataset(
    mesh_dir="/path/to/meshes",
    label_dir="/path/to/labels", 
    texture_dir="/path/to/textures",
    n_clusters=16,
    clusters_per_batch=4,
    PE=True,
    texture_patch_size=16
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=2, collate_fn=texture_custom_collate_fn)

# Create model
model = LightweightTextureGeometryModel(
    geometry_feature_dim=27,  # Adjust based on your features
    texture_embed_dim=64,
    output_dim=256,
    num_attention_blocks=2
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        padded_batches, padded_labels, padded_textures, masks, texture_masks = batch
        
        # Forward pass
        output = model(padded_batches, padded_textures, masks, texture_masks)
        
        # Compute loss
        loss = criterion(output, padded_batches)  # Example: reconstruction
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Dataset Parameters

### Required Parameters

- `mesh_dir`: Directory containing mesh files (.obj or .ply)
- `label_dir`: Directory containing label files (.txt)
- `texture_dir`: Directory containing texture files (.pkl)
- `n_clusters`: Number of clusters for K-Means clustering
- `clusters_per_batch`: Number of clusters per batch
- `PE`: Whether to use positional encoding

### Optional Parameters

- `json_dir`: Directory with additional JSON features
- `augmentation`: Augmentation function for meshes
- `transform`: Transform function for samples
- `include_normals`: Whether to include face normals (default: True)
- `additional_geometrical_features`: Whether to include slope, height, roughness (default: False)
- `texture_patch_size`: Size to resize texture patches to (default: 16)
- `max_texture_pixels`: Maximum pixels per face for padding/truncation (default: 256)

## Data Structure

### Input Shapes

- `padded_batches`: `[B, P, S, F]` - Batched face features
  - B: Batch size
  - P: Number of clusters per batch
  - S: Maximum sequence length (faces per cluster)
  - F: Feature dimension

- `padded_labels`: `[B, P, L]` - Batched labels
  - L: Maximum label sequence length

- `padded_textures`: `[B, P, S, T, C]` - Batched texture sequences
  - T: Maximum texture sequence length (max_texture_pixels)
  - C: Number of channels (3 for RGB)

- `masks`: `[B, P, S]` - Geometry validity masks
- `texture_masks`: `[B, P, S]` - Texture validity masks

### Caching

The dataset automatically caches processed data to speed up subsequent loads:

- Cache files are stored in `{mesh_dir}/.cluster_texture_cache/`
- Cache includes both geometry and texture data
- Cache is invalidated when parameters change
- Augmentation disables caching (computed on-the-fly)

## Texture Processing

### Automatic Processing

The dataset automatically:

1. **Loads texture data** from .pkl files
2. **Validates format** and handles errors gracefully
3. **Processes pixel sequences** as 1D sequences (no 2D reshaping)
4. **Pads/truncates sequences** to standard length (max_texture_pixels)
5. **Handles empty textures** with zero-filled sequences
6. **Creates validity masks** for texture data

### Texture Sequence Processing

- Pixel sequences are padded or truncated to `max_texture_pixels` length
- Empty textures (0 pixels) become zero-filled sequences
- Invalid textures are replaced with zero-filled sequences
- Sequences maintain their 1D structure (no 2D reshaping)

## Error Handling

### Common Issues

1. **Missing texture files**: Creates zero-filled patches
2. **Invalid texture format**: Replaces with zero-filled patches
3. **Empty textures**: Creates zero-filled patches
4. **Corrupted cache**: Automatically recomputes and recaches

### Debugging

```python
# Check texture data loading
dataset = MeshTextureDataset(...)
texture_data, texture_masks = dataset.load_texture_data("mesh_name")
print(f"Loaded {len(texture_data)} texture patches")
print(f"Valid textures: {sum(texture_masks)}")

# Check data shapes
for batch in dataloader:
    padded_batches, padded_labels, padded_textures, masks, texture_masks = batch
    print(f"Textures shape: {padded_textures.shape}")
    print(f"Texture masks: {texture_masks.sum()}")
    break
```

## Data Conversion

### Converting Existing Data

Use the `texture_data_converter.py` script to convert your existing texture data:

```python
from texture_data_converter import convert_texture_data_to_pkl_format

# Convert your texture data
convert_texture_data_to_pkl_format(
    input_dir="/path/to/your/texture/files",
    output_dir="/path/to/converted/textures",
    mesh_dir="/path/to/your/mesh/files"
)
```

### Supported Input Formats

- **Text files**: One face per line with pixel data
- **NumPy arrays**: Shape (num_faces, num_pixels, 3)
- **Binary files**: Custom binary format
- **Other formats**: Extend `load_texture_data()` function

### Creating Sample Data

```python
from texture_data_converter import create_sample_texture_data

# Create sample data for testing
create_sample_texture_data("./sample_textures", num_faces=100)
```

## Performance Tips

### Memory Optimization

1. **Reduce texture_patch_size** for lower memory usage
2. **Decrease max_texture_pixels** to limit patch size
3. **Use smaller batch sizes** if memory is limited
4. **Enable caching** to avoid recomputation

### Speed Optimization

1. **Use caching** for repeated access
2. **Disable augmentation** during validation
3. **Use appropriate num_workers** for DataLoader
4. **Pre-process textures** to standard sizes

### Example Configuration

```python
# Memory-constrained setup
dataset = MeshTextureDataset(
    mesh_dir="...",
    label_dir="...",
    texture_dir="...",
    n_clusters=8,           # Fewer clusters
    clusters_per_batch=2,   # Smaller batches
    texture_patch_size=8,   # Smaller patches
    max_texture_pixels=64   # Fewer pixels
)

# High-performance setup
dataset = MeshTextureDataset(
    mesh_dir="...",
    label_dir="...",
    texture_dir="...",
    n_clusters=32,          # More clusters
    clusters_per_batch=8,   # Larger batches
    texture_patch_size=32,  # Larger patches
    max_texture_pixels=512  # More pixels
)
```

## Integration with Models

### With Lightweight Models

```python
from lightweight_texture_processor import LightweightTextureGeometryModel

model = LightweightTextureGeometryModel(
    geometry_feature_dim=27,
    texture_embed_dim=64,
    output_dim=256,
    fusion_method='gated'
)

# Forward pass
output = model(geometry_features, texture_patches, masks, texture_masks)
```

### With Existing Models

```python
from texture_integration_example import create_texture_aware_model

# Wrap existing model
model = create_texture_aware_model(
    original_model_path='your_model.pth',
    texture_embed_dim=64,
    fusion_method='gated'
)

# Use with texture data
output = model(geometry_data, masks, texture_patches, texture_masks)
```

## Troubleshooting

### Common Problems

1. **"No texture file found"**: Check file naming and directory structure
2. **"Invalid texture format"**: Verify .pkl file structure
3. **"Out of memory"**: Reduce batch size or texture patch size
4. **"Cache corruption"**: Delete cache directory and recompute

### Debug Commands

```python
# Check dataset size
print(f"Dataset length: {len(dataset)}")

# Check texture files
texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.pkl')]
print(f"Found {len(texture_files)} texture files")

# Validate texture format
from texture_data_converter import validate_pkl_format
validate_pkl_format("path/to/texture.pkl")

# Check memory usage
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

## File Structure

```
project/
├── mesh_texture_dataset.py      # Main dataset class
├── texture_data_converter.py    # Data conversion utilities
├── texture_training_example.py  # Training example
├── lightweight_texture_processor.py  # Lightweight models
├── texture_transformer.py       # Full-featured models
└── texture_integration_example.py   # Integration helpers
```

## Next Steps

1. **Convert your texture data** using the converter script
2. **Test data loading** with the sample data
3. **Choose a model size** based on your requirements
4. **Start training** with the example script
5. **Monitor performance** and adjust parameters as needed

For more advanced usage, see the individual model documentation and examples.
