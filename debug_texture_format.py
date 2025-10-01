#!/usr/bin/env python3
"""
Debug script to examine texture data format and identify issues.
"""

import pickle
import os
import sys

def examine_texture_file(texture_file_path, max_faces_to_check=10):
    """
    Examine a texture file to understand its format.
    
    Args:
        texture_file_path (str): Path to the texture .pkl file
        max_faces_to_check (int): Maximum number of faces to examine in detail
    """
    print(f"Examining texture file: {texture_file_path}")
    print("=" * 60)
    
    if not os.path.exists(texture_file_path):
        print(f"File not found: {texture_file_path}")
        return
    
    try:
        with open(texture_file_path, 'rb') as f:
            texture_data = pickle.load(f)
        
        print(f"✓ Successfully loaded texture data")
        print(f"Data type: {type(texture_data)}")
        
        if isinstance(texture_data, list):
            print(f"Number of faces: {len(texture_data)}")
            
            # Examine first few faces in detail
            valid_faces = 0
            invalid_faces = 0
            
            for face_idx in range(min(len(texture_data), max_faces_to_check)):
                face_texture = texture_data[face_idx]
                print(f"\nFace {face_idx}:")
                print(f"  Type: {type(face_texture)}")
                
                if isinstance(face_texture, tuple) and len(face_texture) == 2:
                    num_pixels, pixel_list = face_texture
                    print(f"  ✓ Valid format: tuple with 2 elements")
                    print(f"  num_pixels: {num_pixels} (type: {type(num_pixels)})")
                    print(f"  pixel_list type: {type(pixel_list)}")
                    print(f"  pixel_list length: {len(pixel_list) if hasattr(pixel_list, '__len__') else 'N/A'}")
                    
                    if len(pixel_list) > 0:
                        print(f"  First pixel: {pixel_list[0] if hasattr(pixel_list, '__getitem__') else 'N/A'}")
                        if len(pixel_list) > 1:
                            print(f"  Second pixel: {pixel_list[1] if hasattr(pixel_list, '__getitem__') else 'N/A'}")
                    
                    valid_faces += 1
                    
                else:
                    print(f"  ✗ Invalid format!")
                    print(f"  Expected: tuple with 2 elements")
                    print(f"  Got: {type(face_texture)} with {len(face_texture) if hasattr(face_texture, '__len__') else 'N/A'} elements")
                    
                    if hasattr(face_texture, '__len__') and len(face_texture) > 0:
                        print(f"  First element: {face_texture[0] if hasattr(face_texture, '__getitem__') else 'N/A'}")
                    
                    invalid_faces += 1
            
            print(f"\nSummary for first {min(len(texture_data), max_faces_to_check)} faces:")
            print(f"  Valid faces: {valid_faces}")
            print(f"  Invalid faces: {invalid_faces}")
            
            # Check if there are more invalid faces beyond what we examined
            if len(texture_data) > max_faces_to_check:
                additional_invalid = 0
                for face_idx in range(max_faces_to_check, len(texture_data)):
                    face_texture = texture_data[face_idx]
                    if not (isinstance(face_texture, tuple) and len(face_texture) == 2):
                        additional_invalid += 1
                
                if additional_invalid > 0:
                    print(f"  Additional invalid faces (not examined): {additional_invalid}")
            
        else:
            print(f"✗ Texture data is not a list: {type(texture_data)}")
            
    except Exception as e:
        print(f"✗ Error loading texture file: {e}")

def find_problematic_texture_files(texture_dir, max_files=5):
    """
    Find and examine texture files that might have format issues.
    """
    print(f"Scanning texture directory: {texture_dir}")
    print("=" * 60)
    
    if not os.path.exists(texture_dir):
        print(f"Directory not found: {texture_dir}")
        return
    
    texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.pkl')]
    print(f"Found {len(texture_files)} texture files")
    
    if len(texture_files) == 0:
        print("No texture files found!")
        return
    
    # Examine a few files
    for i, texture_file in enumerate(texture_files[:max_files]):
        texture_path = os.path.join(texture_dir, texture_file)
        print(f"\n{'='*20} File {i+1}/{min(len(texture_files), max_files)} {'='*20}")
        examine_texture_file(texture_path)
        
        if i < len(texture_files) - 1:
            input("\nPress Enter to continue to next file...")

def suggest_fixes():
    """Suggest potential fixes for texture format issues."""
    print("\n" + "="*60)
    print("SUGGESTED FIXES")
    print("="*60)
    
    print("""
The warnings indicate that some faces have texture data in an unexpected format.

Expected format for each face:
- A tuple with exactly 2 elements: (num_pixels, pixel_list)
- num_pixels: integer indicating number of pixels
- pixel_list: list of RGB values like [[R,G,B], [R,G,B], ...]

Possible issues and fixes:

1. **Different data structure**: Some faces might have texture data in a different format
   - Fix: Add more flexible parsing in load_texture_data()

2. **Missing or corrupted data**: Some faces might have None, empty, or corrupted data
   - Fix: Add better error handling and fallback to zero textures

3. **Different tuple structure**: Some faces might have tuples with different lengths
   - Fix: Handle various tuple lengths gracefully

4. **Non-tuple data**: Some faces might have texture data as lists or other types
   - Fix: Convert to expected format if possible

Would you like me to implement a more robust texture loading function?
""")

if __name__ == "__main__":
    print("Texture Format Debug Tool")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Examine specific file
        texture_file = sys.argv[1]
        examine_texture_file(texture_file)
    else:
        # Interactive mode
        print("This tool helps debug texture data format issues.")
        print("\nOptions:")
        print("1. Examine a specific texture file")
        print("2. Scan a texture directory")
        print("3. Show suggested fixes")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            texture_file = input("Enter path to texture .pkl file: ").strip()
            examine_texture_file(texture_file)
        elif choice == "2":
            texture_dir = input("Enter path to texture directory: ").strip()
            find_problematic_texture_files(texture_dir)
        elif choice == "3":
            suggest_fixes()
        else:
            print("Invalid choice!")
    
    suggest_fixes()
