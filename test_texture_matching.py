#!/usr/bin/env python3
"""
Test script to verify texture file matching logic for the specific dataset.
"""

import os
import sys
from mesh_texture_dataset import MeshTextureDataset, get_texture_base_name

def test_texture_matching():
    """Test the texture file matching logic."""
    
    # Dataset paths
    mesh_dir = "/bigwork/nhgnheid/data_Dlft/trian_mesh"
    label_dir = "/bigwork/nhgnheid/data_Dlft/train_label"
    texture_dir = "/bigwork/nhgnheid/data_Dlft/train_texture"
    
    # Check if directories exist
    if not all(os.path.exists(d) for d in [mesh_dir, label_dir, texture_dir]):
        print("Error: One or more directories don't exist")
        return False
    
    print("Testing texture file matching...")
    print(f"Mesh dir: {mesh_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Texture dir: {texture_dir}")
    print()
    
    # Test specific example
    test_mesh = "Tile_+1984_+2692_groundtruth_L1.ply"
    expected_label = "Tile_+1984_+2692_groundtruth_L1.txt"
    expected_texture = "Tile_+1984_+2692_groundtruth_L1_pixels_test.pkl"
    
    print(f"Testing with mesh file: {test_mesh}")
    
    # Check if files exist
    mesh_path = os.path.join(mesh_dir, test_mesh)
    label_path = os.path.join(label_dir, expected_label)
    texture_path = os.path.join(texture_dir, expected_texture)
    
    print(f"Mesh file exists: {os.path.exists(mesh_path)}")
    print(f"Label file exists: {os.path.exists(label_path)}")
    print(f"Texture file exists: {os.path.exists(texture_path)}")
    print()
    
    # Test the texture base name function
    base_name = os.path.splitext(test_mesh)[0]  # "Tile_+1984_+2692_groundtruth_L1"
    texture_base_name = get_texture_base_name(base_name)
    
    print(f"Mesh base name: {base_name}")
    print(f"Texture base name: {texture_base_name}")
    
    # Test the texture file lookup logic
    standard_texture_path = os.path.join(texture_dir, texture_base_name + '.pkl')
    alt_texture_path = os.path.join(texture_dir, texture_base_name + '_pixels_test.pkl')
    
    print(f"Standard texture path: {standard_texture_path}")
    print(f"Standard texture exists: {os.path.exists(standard_texture_path)}")
    print(f"Alternative texture path: {alt_texture_path}")
    print(f"Alternative texture exists: {os.path.exists(alt_texture_path)}")
    print()
    
    # Try to create the dataset (this will test the actual matching logic)
    try:
        print("Creating MeshTextureDataset to test matching...")
        dataset = MeshTextureDataset(
            mesh_dir=mesh_dir,
            label_dir=label_dir,
            texture_dir=texture_dir,
            n_clusters=10,
            clusters_per_batch=2,
            PE=True
        )
        print("✓ Dataset created successfully!")
        print(f"Dataset reports {len(dataset.mesh_files)} mesh files")
        print("✓ Texture matching logic is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_texture_matching()
    sys.exit(0 if success else 1)





