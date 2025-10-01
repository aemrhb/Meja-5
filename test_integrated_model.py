#!/usr/bin/env python3
"""
Test script for the integrated texture-geometry model.
This script tests the model with synthetic data to ensure everything works correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrated_texture_geometry_model import (
    IntegratedTextureGeometryModel, 
    IntegratedDownstreamClassifier,
    create_integrated_model
)

def test_integrated_model():
    """Test the integrated texture-geometry model with synthetic data."""
    print("Testing Integrated Texture-Geometry Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    geometry_feature_dim = 27  # 9 features per vertex * 3 vertices
    embedding_dim = 256
    texture_embed_dim = 128
    num_heads = 4
    num_attention_blocks = 4
    num_classes = 10
    max_texture_pixels = 256
    fusion_method = 'gated'
    
    # Create model
    print("\n1. Creating integrated model...")
    model = create_integrated_model(
        geometry_feature_dim=geometry_feature_dim,
        embedding_dim=embedding_dim,
        texture_embed_dim=texture_embed_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        dropout=0.1,
        summary_mode='cls',
        use_hierarchical=False,
        fourier=False,
        relative_positional_encoding=False,
        fusion_method=fusion_method,
        max_texture_pixels=max_texture_pixels
    ).to(device)
    
    print(f"   Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test data dimensions
    B, N, T = 2, 8, 10  # batch, clusters, faces per cluster
    H, W, C = max_texture_pixels, 3, 3  # texture dimensions
    
    print(f"\n2. Creating test data...")
    print(f"   Batch size: {B}")
    print(f"   Clusters per batch: {N}")
    print(f"   Faces per cluster: {T}")
    print(f"   Texture sequence length: {H}")
    
    # Create synthetic data
    geometry_features = torch.randn(B, N, T, geometry_feature_dim).to(device)
    texture_sequences = torch.randint(0, 255, (B, N, H, C)).to(device).float()
    masks = torch.ones(B, N, T).to(device)
    texture_masks = torch.ones(B, N).to(device)
    
    print(f"   Geometry features shape: {geometry_features.shape}")
    print(f"   Texture sequences shape: {texture_sequences.shape}")
    print(f"   Geometry masks shape: {masks.shape}")
    print(f"   Texture masks shape: {texture_masks.shape}")
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(geometry_features, texture_sequences, masks, texture_masks)
            print(f"   ✓ Forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected shape: ({B}, {N}, {T}, {embedding_dim})")
            
            if output.shape == (B, N, T, embedding_dim):
                print(f"   ✓ Output shape matches expected")
            else:
                print(f"   ✗ Output shape mismatch!")
                return False
                
        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            return False
    
    # Test downstream classifier
    print(f"\n4. Testing downstream classifier...")
    classifier = IntegratedDownstreamClassifier(
        integrated_encoder=model,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout=0.1,
        freeze_encoder_layers=2,
        fusion_method=fusion_method
    ).to(device)
    
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"   Classifier parameters: {classifier_params:,}")
    
    with torch.no_grad():
        try:
            logits = classifier(geometry_features, texture_sequences, masks, texture_masks)
            print(f"   ✓ Classifier forward pass successful")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Expected shape: ({B}, {N}, {T}, {num_classes})")
            
            if logits.shape == (B, N, T, num_classes):
                print(f"   ✓ Classifier output shape matches expected")
            else:
                print(f"   ✗ Classifier output shape mismatch!")
                return False
                
        except Exception as e:
            print(f"   ✗ Classifier forward pass failed: {e}")
            return False
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    model.train()
    classifier.train()
    
    try:
        # Forward pass
        embeddings = model(geometry_features, texture_sequences, masks, texture_masks)
        logits = classifier(geometry_features, texture_sequences, masks, texture_masks)
        
        # Create dummy loss
        target = torch.randint(0, num_classes, (B, N, T)).to(device)
        loss = nn.CrossEntropyLoss()(logits.view(-1, num_classes), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        print(f"   ✓ Gradient flow successful")
        print(f"   Loss value: {loss.item():.4f}")
        
        # Check if gradients exist
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print(f"   ✓ Gradients computed successfully")
        else:
            print(f"   ✗ No gradients found!")
            return False
            
    except Exception as e:
        print(f"   ✗ Gradient flow failed: {e}")
        return False
    
    # Test different fusion methods
    print(f"\n6. Testing different fusion methods...")
    fusion_methods = ['gated', 'concat', 'add']
    
    for method in fusion_methods:
        try:
            test_model = create_integrated_model(
                geometry_feature_dim=geometry_feature_dim,
                embedding_dim=embedding_dim,
                texture_embed_dim=texture_embed_dim,
                num_heads=num_heads,
                num_attention_blocks=2,  # Fewer blocks for faster testing
                dropout=0.1,
                fusion_method=method,
                max_texture_pixels=max_texture_pixels
            ).to(device)
            
            with torch.no_grad():
                output = test_model(geometry_features, texture_sequences, masks, texture_masks)
                print(f"   ✓ Fusion method '{method}' works")
                
        except Exception as e:
            print(f"   ✗ Fusion method '{method}' failed: {e}")
            return False
    
    print(f"\n" + "=" * 50)
    print(f"✓ All tests passed! The integrated model is working correctly.")
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with empty texture masks
    print("\n1. Testing with empty texture masks...")
    try:
        model = create_integrated_model(
            geometry_feature_dim=27,
            embedding_dim=256,
            texture_embed_dim=128,
            fusion_method='gated'
        ).to(device)
        
        B, N, T = 1, 2, 5
        geometry_features = torch.randn(B, N, T, 27).to(device)
        texture_sequences = torch.randint(0, 255, (B, N, 256, 3)).to(device).float()
        masks = torch.ones(B, N, T).to(device)
        texture_masks = torch.zeros(B, N).to(device)  # All textures invalid
        
        with torch.no_grad():
            output = model(geometry_features, texture_sequences, masks, texture_masks)
            print(f"   ✓ Empty texture masks handled correctly")
            
    except Exception as e:
        print(f"   ✗ Empty texture masks failed: {e}")
        return False
    
    # Test with different batch sizes
    print("\n2. Testing with different batch sizes...")
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            B, N, T = batch_size, 4, 6
            geometry_features = torch.randn(B, N, T, 27).to(device)
            texture_sequences = torch.randint(0, 255, (B, N, 256, 3)).to(device).float()
            masks = torch.ones(B, N, T).to(device)
            texture_masks = torch.ones(B, N).to(device)
            
            with torch.no_grad():
                output = model(geometry_features, texture_sequences, masks, texture_masks)
                print(f"   ✓ Batch size {batch_size} works")
                
        except Exception as e:
            print(f"   ✗ Batch size {batch_size} failed: {e}")
            return False
    
    print(f"\n✓ All edge case tests passed!")
    return True

if __name__ == "__main__":
    print("Integrated Texture-Geometry Model Test Suite")
    print("=" * 60)
    
    # Run main tests
    success = test_integrated_model()
    
    if success:
        # Run edge case tests
        success = test_edge_cases()
    
    if success:
        print(f"\n🎉 All tests passed! The model is ready for training.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
