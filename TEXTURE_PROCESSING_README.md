# Lightweight Texture Processing for 3D Mesh Analysis

This repository contains lightweight transformer-based models for processing raw pixel textures and integrating them with geometric features for 3D mesh analysis.

## Overview

The models are designed to:
1. **Process raw pixel textures** from mesh faces into compact embeddings
2. **Integrate texture information** with existing geometric features
3. **Minimize computational overhead** while maintaining performance
4. **Provide flexible fusion strategies** for different use cases

## Models

### 1. `texture_transformer.py` - Full-Featured Texture Processor

**Features:**
- CNN-based patch embedding with configurable patch size
- Multi-head attention with CLS token or average pooling
- Hierarchical processing with local and global attention
- Integration with existing geometry models

**Key Components:**
- `PatchEmbedding`: Converts raw pixels to embeddings using lightweight CNN
- `LightweightMultiHeadAttention`: Efficient attention mechanism
- `TextureSummarizer`: Main texture processing module
- `IntegratedTextureGeometryModel`: Complete fusion model

**Usage:**
```python
from texture_transformer import IntegratedTextureGeometryModel

model = IntegratedTextureGeometryModel(
    geometry_feature_dim=27,
    texture_patch_size=16,
    texture_channels=3,
    embed_dim=256,
    texture_embed_dim=128,
    num_heads=4,
    num_texture_layers=3,
    num_attention_blocks=4
)

# Forward pass
output = model(geometry_features, texture_patches, masks, texture_masks)
```

### 2. `lightweight_texture_processor.py` - Ultra-Lightweight Version

**Features:**
- Minimal parameter count (32-64 embedding dimensions)
- Depthwise separable convolutions for efficiency
- Shared QKV projections to reduce parameters
- Multiple fusion strategies (concat, add, gated)

**Key Components:**
- `UltraLightTextureProcessor`: Minimal texture processor
- `EfficientTextureGeometryFusion`: Lightweight fusion module
- `LightweightTextureGeometryModel`: Complete lightweight model

**Usage:**
```python
from lightweight_texture_processor import LightweightTextureGeometryModel

model = LightweightTextureGeometryModel(
    geometry_feature_dim=27,
    texture_embed_dim=64,  # Very small
    output_dim=256,
    num_attention_blocks=2,
    fusion_method='gated'  # 'concat', 'add', 'gated'
)
```

### 3. `texture_integration_example.py` - Integration Helper

**Features:**
- Wrapper for existing models to add texture processing
- Gradual integration without major code changes
- Training helpers and loss functions
- Multiple fusion strategies

**Usage:**
```python
from texture_integration_example import create_texture_aware_model

# Wrap your existing model
model = create_texture_aware_model(
    original_model_path='path/to/your/model.pth',
    texture_embed_dim=64,
    fusion_method='gated'
)

# Use in training
output = model(geometry_data, masks, texture_patches, texture_masks)
```

## Model Comparison

| Model | Parameters | Texture Embed Dim | Attention Blocks | Use Case |
|-------|------------|-------------------|------------------|----------|
| Ultra Light | ~50K | 32 | 0 (MLP only) | Mobile/Edge devices |
| Light | ~200K | 64 | 2 | Balanced performance |
| Medium | ~500K | 128 | 4 | High accuracy needs |
| Full | ~1M+ | 128+ | 4+ | Research/High-end |

## Integration Strategies

### 1. Early Fusion (Recommended)
Process texture and geometry separately, then fuse before main processing:
```python
# Process texture
texture_emb = texture_processor(texture_patches)

# Process geometry  
geometry_emb = geometry_processor(geometry_features)

# Fuse
fused = fusion_layer(geometry_emb, texture_emb)
```

### 2. Late Fusion
Process through main model, then add texture information:
```python
# Main processing
main_output = main_model(geometry_features)

# Add texture
texture_emb = texture_processor(texture_patches)
enhanced_output = main_output + texture_emb
```

### 3. Gated Fusion
Use learned gates to control texture vs geometry contribution:
```python
gate = sigmoid(linear([geometry_emb, texture_emb]))
output = gate * geometry_emb + (1 - gate) * texture_emb
```

## Training Tips

### 1. Data Preparation
```python
# Extract texture patches from mesh faces
def extract_texture_patches(mesh_faces, texture_image, patch_size=16):
    patches = []
    for face in mesh_faces:
        # Get UV coordinates for face
        uv_coords = get_face_uv_coordinates(face)
        
        # Sample texture patch
        patch = sample_texture_patch(texture_image, uv_coords, patch_size)
        patches.append(patch)
    
    return torch.stack(patches)
```

### 2. Loss Functions
```python
def texture_geometry_loss(pred, target, texture_embeddings=None):
    # Main task loss
    main_loss = F.mse_loss(pred, target)
    
    # Optional texture consistency loss
    if texture_embeddings is not None:
        texture_loss = F.mse_loss(texture_embeddings, texture_embeddings.detach())
        return main_loss + 0.1 * texture_loss
    
    return main_loss
```

### 3. Gradual Integration
Start with geometry-only, then gradually increase texture contribution:
```python
# Phase 1: Geometry only
texture_weight = 0.0

# Phase 2: Add texture gradually
texture_weight = 0.1

# Phase 3: Full texture integration
texture_weight = 0.3
```

## Performance Optimization

### 1. Memory Efficiency
- Use smaller patch sizes (8x8 instead of 16x16)
- Reduce embedding dimensions
- Use gradient checkpointing for large models

### 2. Speed Optimization
- Use depthwise separable convolutions
- Reduce attention heads
- Use MLP blocks instead of attention for ultra-light models

### 3. Quality vs Speed Trade-offs
```python
# Ultra-fast (mobile)
model = LightweightTextureGeometryModel(
    texture_embed_dim=32,
    use_attention=False,
    fusion_method='add'
)

# Balanced
model = LightweightTextureGeometryModel(
    texture_embed_dim=64,
    num_attention_blocks=2,
    fusion_method='gated'
)

# High quality
model = LightweightTextureGeometryModel(
    texture_embed_dim=128,
    num_attention_blocks=4,
    fusion_method='concat'
)
```

## Example Training Loop

```python
import torch
import torch.optim as optim
from lightweight_texture_processor import LightweightTextureGeometryModel

# Create model
model = LightweightTextureGeometryModel(
    geometry_feature_dim=27,
    texture_embed_dim=64,
    output_dim=256,
    num_attention_blocks=2
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        geometry_features, texture_patches, masks, targets = batch
        
        # Forward pass
        output = model(geometry_features, texture_patches, masks)
        
        # Loss
        loss = F.mse_loss(output, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `texture_embed_dim` or `patch_size`
2. **Poor Texture Quality**: Increase `num_texture_layers` or `texture_embed_dim`
3. **Slow Training**: Use `use_attention=False` or reduce `num_attention_blocks`
4. **Texture Not Helping**: Try different `fusion_method` or increase `texture_weight`

### Debugging:
```python
# Check texture embeddings
with torch.no_grad():
    texture_emb = model.texture_processor(texture_patches)
    print(f"Texture embedding stats: mean={texture_emb.mean():.4f}, std={texture_emb.std():.4f}")

# Check fusion weights
if hasattr(model.fusion, 'gate'):
    gate_weights = model.fusion.gate(combined_features)
    print(f"Gate weights: {gate_weights.mean():.4f}")
```

## Future Improvements

1. **Adaptive Patch Sizing**: Use different patch sizes based on face area
2. **Multi-Scale Processing**: Process textures at multiple resolutions
3. **Attention Visualization**: Visualize which texture regions are important
4. **Quantization**: INT8 quantization for mobile deployment
5. **Neural Architecture Search**: Automatically find optimal architectures

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lightweight_texture_processing,
  title={Lightweight Transformer-based Texture Processing for 3D Mesh Analysis},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository}
}
```
