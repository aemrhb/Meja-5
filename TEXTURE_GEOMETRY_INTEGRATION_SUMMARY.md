# Texture-Geometry Integration Summary

This document summarizes the changes made to support training with both texture and geometry features.

## Overview

The training script has been modified to support both texture and geometry features, allowing the model to learn from both visual appearance (texture) and 3D structure (geometry) of mesh faces.

## Key Changes Made

### 1. Integrated Model Architecture (`integrated_texture_geometry_model.py`)

**New Components:**
- `TexturePatchEmbedding`: Processes raw texture pixel sequences using 1D convolutions
- `TextureGeometryFusion`: Fuses texture and geometry embeddings using different strategies (gated, concat, add)
- `IntegratedTextureGeometryModel`: Main model that combines both feature types
- `IntegratedDownstreamClassifier`: Downstream classifier for the integrated model

**Key Features:**
- Handles texture sequences (not 2D patches) using 1D convolutions
- Multiple fusion strategies: gated, concatenation, and addition
- Compatible with existing geometry-only models
- Supports layer-wise freezing for fine-tuning

### 2. Enhanced Texture Dataset (`mesh_texture_dataset.py`)

**Naming Convention Fix:**
- Handles mesh files with `_labeled` suffix (e.g., `51382_542673_20_labeled.obj`)
- Automatically maps to texture files without suffix (e.g., `51382_542673_20.pkl`)
- Added `get_texture_base_name()` helper function

**Robust Texture Format Handling:**
- Supports multiple texture data formats:
  - Standard: `(num_pixels, pixel_list)`
  - Single tuple: `(pixel_list,)`
  - Direct list: `pixel_list`
  - None values and empty data
- Graceful fallback to zero textures for invalid formats
- Better error handling and debugging information

**Custom Collate Function:**
- `texture_custom_collate_fn()` handles varying texture sequence lengths
- Proper padding and masking for both geometry and texture data

### 3. Modified Training Script (`train_downstream_2.py`)

**Configuration Support:**
- New texture-specific configuration parameters:
  ```yaml
  model:
    use_texture: true
    texture_embed_dim: 128
    fusion_method: "gated"  # or "concat", "add"
    max_texture_pixels: 256
    texture_patch_size: 16
  
  paths:
    texture_dir: "/path/to/textures"
    val_texture_dir: "/path/to/val/textures"
  ```

**Dynamic Dataset Selection:**
- Automatically chooses `MeshTextureDataset` when texture features are enabled
- Falls back to `MeshDataset` for geometry-only training
- Uses appropriate collate functions for each dataset type

**Model Architecture Selection:**
- Creates `IntegratedTextureGeometryModel` for texture+geometry training
- Falls back to original `nomeformer` for geometry-only training
- Supports loading pretrained geometry encoder weights

**Training Loop Updates:**
- Handles different input formats for texture vs geometry-only models
- Updated evaluation function to support both model types
- Maintains backward compatibility with existing training pipelines

## Usage Instructions

### 1. Configuration

Create a configuration file with texture support:

```yaml
# config_texture_geometry.yaml
paths:
  mesh_dir: "/path/to/train/meshes"
  label_dir: "/path/to/train/labels"
  texture_dir: "/path/to/train/textures"  # NEW
  val_mesh_dir: "/path/to/val/meshes"
  val_label_dir: "/path/to/val/labels"
  val_texture_dir: "/path/to/val/textures"  # NEW
  # ... other paths

model:
  use_texture: true  # NEW - Enable texture features
  texture_embed_dim: 128  # NEW - Texture embedding dimension
  fusion_method: "gated"  # NEW - How to fuse texture and geometry
  max_texture_pixels: 256  # NEW - Max pixels per texture sequence
  # ... other model parameters

training:
  # ... existing training parameters
```

### 2. Data Organization

**Mesh Files:**
- Can have `_labeled` suffix: `51382_542673_20_labeled.obj`
- Labels: `51382_542673_20_labeled.txt`

**Texture Files:**
- No `_labeled` suffix: `51382_542673_20.pkl`
- Contains texture pixel sequences for each face

### 3. Running Training

```bash
python train_downstream_2.py config_texture_geometry.yaml
```

### 4. Texture Data Format

The texture data should be in `.pkl` files with the following structure:

```python
# Each .pkl file contains a list where each element corresponds to a face
texture_data = [
    (num_pixels_1, [[R, G, B], [R, G, B], ...]),  # Face 0
    (num_pixels_2, [[R, G, B], [R, G, B], ...]),  # Face 1
    # ... more faces
]
```

**Supported Formats:**
- Standard: `(num_pixels, pixel_list)`
- Alternative: `(pixel_list,)` or just `pixel_list`
- Empty: `None` or `[]` (creates zero texture)

## Benefits

### 1. Enhanced Feature Learning
- Combines visual appearance (texture) with geometric structure
- More robust feature representations for complex 3D objects
- Better generalization to diverse mesh appearances

### 2. Flexible Architecture
- Multiple fusion strategies for different use cases
- Configurable texture embedding dimensions
- Layer-wise freezing for efficient fine-tuning

### 3. Robust Data Handling
- Handles various texture data formats gracefully
- Automatic naming convention resolution
- Comprehensive error handling and debugging

### 4. Backward Compatibility
- Existing geometry-only models continue to work
- Gradual migration path from geometry-only to texture+geometry
- No breaking changes to existing training pipelines

## Troubleshooting

### Common Issues

1. **"Invalid texture format" warnings:**
   - The improved format handling should resolve most of these
   - Use `debug_texture_format.py` to examine problematic files
   - Check texture data format matches expected structure

2. **"No texture file found" warnings:**
   - Verify texture directory path in configuration
   - Check naming convention (mesh files with `_labeled`, texture files without)
   - Use `test_naming_convention.py` to verify file matching

3. **Memory issues with large textures:**
   - Reduce `max_texture_pixels` parameter
   - Use smaller `texture_embed_dim`
   - Consider reducing batch size

4. **Poor performance:**
   - Try different fusion methods (`gated`, `concat`, `add`)
   - Adjust `texture_embed_dim` relative to `embedding_dim`
   - Experiment with layer-wise learning rate decay

### Debug Tools

1. **`debug_texture_format.py`**: Examine texture file formats
2. **`test_naming_convention.py`**: Verify file naming and matching
3. **`test_texture_formats.py`**: Test texture format handling
4. **`test_integrated_model.py`**: Test the complete integrated model

## Performance Considerations

### Memory Usage
- Texture sequences require additional memory
- Consider reducing `max_texture_pixels` for large datasets
- Monitor GPU memory usage during training

### Training Speed
- Texture processing adds computational overhead
- 1D convolutions are more efficient than 2D for sequences
- Consider using smaller `texture_embed_dim` for faster training

### Model Size
- Integrated model has more parameters than geometry-only
- Texture components add ~10-20% parameter overhead
- Use layer-wise freezing to reduce trainable parameters

## Future Improvements

1. **Advanced Texture Processing:**
   - Attention-based texture summarization
   - Multi-scale texture features
   - Texture-specific data augmentation

2. **Enhanced Fusion Strategies:**
   - Learned fusion weights
   - Cross-modal attention mechanisms
   - Hierarchical feature fusion

3. **Efficiency Optimizations:**
   - Texture caching and preprocessing
   - Dynamic texture sequence lengths
   - Quantized texture representations
