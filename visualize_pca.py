import argparse
import os
import random
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Local imports
from model_G_2 import nomeformer
from tools.downst import DownstreamClassifier


def _pca_to_rgb(features_np: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X = scaler.fit_transform(features_np)
    pcs3 = PCA(n_components=3, random_state=0).fit_transform(X)
    mins = pcs3.min(axis=0, keepdims=True)
    maxs = pcs3.max(axis=0, keepdims=True)
    denom = np.clip(maxs - mins, 1e-8, None)
    pcs01 = (pcs3 - mins) / denom
    colors = (pcs01 * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.concatenate([colors, 255 * np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
    return rgba


def save_pca_rgb_legend(save_path: str, num_steps: int = 256, z_slices=None, single_z: float = None) -> None:
    """Save an image visualizing the PCA->RGB mapping used here.

    - X axis: PC1 mapped to Red channel (0..1)
    - Y axis: PC2 mapped to Green channel (0..1)
    - If single_z is provided, save a single panel with PC3 fixed to single_z.
    - Otherwise, save multiple rows where PC3 (Blue) is fixed to each value in z_slices.
    """
    x = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    if single_z is not None:
        z = float(np.clip(single_z, 0.0, 1.0))
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        Z = np.full_like(X, fill_value=z)
        rgb = np.stack([X, Y, Z], axis=-1)
        ax.imshow(rgb, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        ax.set_title(f"PCA→RGB (PC3 Blue fixed at {z:.2f})")
        ax.set_xlabel("PC1 → Red")
        ax.set_ylabel("PC2 → Green")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    if z_slices is None:
        z_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

    rows = len(z_slices)
    fig, axes = plt.subplots(rows, 1, figsize=(6, 1.8 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for ax, z in zip(axes, z_slices):
        Z = np.full_like(X, fill_value=z)
        rgb = np.stack([X, Y, Z], axis=-1)
        ax.imshow(rgb, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        ax.set_title(f"PC3 (Blue) = {z:.2f}")
        ax.set_xlabel("PC1 → Red")
        ax.set_ylabel("PC2 → Green")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

    fig.suptitle("PCA to RGB mapping: (R,G,B) = (PC1, PC2, PC3)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_pca_rgb_cube(save_path: str, num_steps: int = 256) -> None:
    """Save a cube-style legend showing multiple faces of the 3D mapping.

    We render these 2D faces of the (PC1,PC2,PC3) -> (R,G,B) cube:
    - XY at Z=0 and Z=1 (bottom/top faces)
    - XZ at Y=1 (front face)
    - YZ at X=1 (right face)
    This gives an overview without fixing to only one slice in the output.
    """
    x = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    faces = []
    labels = []

    # XY at Z=0
    Z0 = np.zeros_like(X)
    faces.append(np.stack([X, Y, Z0], axis=-1))
    labels.append("XY at Z=0 (Blue=0)")

    # XY at Z=1
    Z1 = np.ones_like(X)
    faces.append(np.stack([X, Y, Z1], axis=-1))
    labels.append("XY at Z=1 (Blue=1)")

    # XZ at Y=1
    Y1 = np.ones_like(X)
    faces.append(np.stack([X, Y1, X*0 + X[0,0]], axis=-1))  # placeholder; will rebuild properly below
    # Proper XZ grid: reuse X as X, and Y as Z axis here
    Z = Y  # reuse meshgrid second axis as Z for display
    faces[-1] = np.stack([X, Y1, Z], axis=-1)
    labels.append("XZ at Y=1 (Green=1)")

    # YZ at X=1
    X1 = np.ones_like(X)
    faces.append(np.stack([X1, X, Y], axis=-1))  # here use meshgrid X as Green, Y as Blue
    labels.append("YZ at X=1 (Red=1)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, rgb, title in zip(axes, faces, labels):
        ax.imshow(rgb, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
    fig.suptitle("PCA→RGB color cube faces")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _write_ascii_ply_with_colors(file_path: str, vertices_xyz: np.ndarray, faces_idx: np.ndarray, vertex_colors_rgb: np.ndarray, face_colors_rgb: np.ndarray) -> None:
    """Write an ASCII PLY with per-vertex and per-face RGB colors.

    vertices_xyz: (N, 3) float32/float64 positions
    faces_idx:    (M, 3) int vertex indices (triangles)
    vertex_colors_rgb: (N, 3) uint8 vertex colors (r,g,b)
    face_colors_rgb:   (M, 3) uint8 face colors (r,g,b)
    """
    num_vertices = int(vertices_xyz.shape[0])
    num_faces = int(faces_idx.shape[0])

    with open(file_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Vertices
        for i in range(num_vertices):
            x, y, z = vertices_xyz[i]
            r, g, b = vertex_colors_rgb[i]
            f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")

        # Faces (assume triangles)
        for j in range(num_faces):
            v0, v1, v2 = faces_idx[j]
            r, g, b = face_colors_rgb[j]
            f.write(f"3 {int(v0)} {int(v1)} {int(v2)} {int(r)} {int(g)} {int(b)}\n")


def _write_ascii_ply_with_labels(file_path: str, vertices_xyz: np.ndarray, faces_idx: np.ndarray, vertex_labels: np.ndarray, face_labels: np.ndarray) -> None:
    """Write an ASCII PLY with per-vertex and per-face classification labels.

    vertices_xyz: (N, 3) float32/float64 positions
    faces_idx:    (M, 3) int vertex indices (triangles)
    vertex_labels: (N,) int vertex classification labels
    face_labels:   (M,) int face classification labels
    """
    num_vertices = int(vertices_xyz.shape[0])
    num_faces = int(faces_idx.shape[0])

    with open(file_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int label\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property int label\n")
        f.write("end_header\n")

        # Vertices
        for i in range(num_vertices):
            x, y, z = vertices_xyz[i]
            label = int(vertex_labels[i])
            f.write(f"{float(x)} {float(y)} {float(z)} {label}\n")

        # Faces (assume triangles)
        for j in range(num_faces):
            v0, v1, v2 = faces_idx[j]
            label = int(face_labels[j])
            f.write(f"3 {int(v0)} {int(v1)} {int(v2)} {label}\n")


def _report_duplicate_faces(mesh: trimesh.Trimesh) -> int:
    """Report number of duplicate face groups (same vertex set, any winding)."""
    faces = mesh.faces
    if faces.size == 0:
        return 0
    faces_norm = np.sort(faces, axis=1)
    _, counts = np.unique(faces_norm, axis=0, return_counts=True)
    num_dup_groups = int((counts > 1).sum())
    if num_dup_groups > 0:
        total_dups = int((counts[counts > 1] - 1).sum())
        print(f"Detected {num_dup_groups} duplicate face groups ({total_dups} duplicate face entries).")
    else:
        print("No duplicate faces detected.")
    return num_dup_groups


def _propagate_colors_to_duplicate_faces(mesh: trimesh.Trimesh, rgba: np.ndarray) -> np.ndarray:
    """Copy a representative color to all duplicates so duplicate faces share identical colors."""
    faces = mesh.faces
    if faces.size == 0:
        return rgba
    faces_norm = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(faces_norm, axis=0, return_inverse=True, return_counts=True)
    if not np.any(counts > 1):
        return rgba
    # group indices by group id
    group_to_indices = {}
    for face_index, group_id in enumerate(inverse):
        gid = int(group_id)
        if gid not in group_to_indices:
            group_to_indices[gid] = []
        group_to_indices[gid].append(int(face_index))
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            continue
        ref = indices[0]
        rgba[indices] = rgba[ref]
    return rgba


def _propagate_labels_to_duplicate_faces(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    """Copy a representative label to all duplicates so duplicate faces share identical labels."""
    faces = mesh.faces
    if faces.size == 0:
        return labels
    faces_norm = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(faces_norm, axis=0, return_inverse=True, return_counts=True)
    if not np.any(counts > 1):
        return labels
    # group indices by group id
    group_to_indices = {}
    for face_index, group_id in enumerate(inverse):
        gid = int(group_id)
        if gid not in group_to_indices:
            group_to_indices[gid] = []
        group_to_indices[gid].append(int(face_index))
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            continue
        ref = indices[0]
        labels[indices] = labels[ref]
    return labels

@torch.no_grad()
def visualize_random_mesh_pca(
    target_encoder: torch.nn.Module,
    mesh_dir: str,
    save_root: str,
    device: torch.device,
    faces_per_cluster: int,
    clusters_per_batch: int,
    use_pe: bool,
    n_clusters_fixed: int = None,
    mesh_path: str = None,
    save_ascii: bool = True,
    classifier: torch.nn.Module = None,
    num_classes: int = None,
):
    os.makedirs(save_root, exist_ok=True)

    if mesh_path is None:
        print('mesj dir', mesh_dir)
        candidates = [f for f in os.listdir(mesh_dir) if f.endswith((".obj", ".ply", ".off"))]
        if not candidates:
            print("No mesh files found for PCA visualization.")
            return
        fname = random.choice(candidates)
        mesh_path = os.path.join(mesh_dir, fname)
    else:
        fname = os.path.basename(mesh_path)
    mesh = trimesh.load(mesh_path, force='mesh')
    _report_duplicate_faces(mesh)

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

    # If a specific mesh file is requested, restrict the dataset to that file only
    if mesh_path is not None:
        vis_ds.index_map = [entry for entry in vis_ds.index_map if entry[0] == mesh_path]
        if len(vis_ds.index_map) == 0:
            print(f"Requested mesh not indexed by dataset: {mesh_path}")
            return

    num_faces = mesh.faces.shape[0]
    feature_dim_accum = None
    face_features = [None] * num_faces
    print('len(vis_ds)', len(vis_ds))
    for idx in range(len(vis_ds)):
        clusters, cluster_face_indices, cluster_face_ids = vis_ds[idx]
        max_seq = max(cl.size(0) for cl in clusters)
        feat_dim = clusters[0].size(1)
        if feature_dim_accum is None:
            feature_dim_accum = feat_dim
        padded_data = [F.pad(cl, (0, 0, 0, max_seq - cl.size(0))) for cl in clusters]
        batch_tensor = torch.stack(padded_data).unsqueeze(0).to(device)

        enc_masks = []
        for cl in clusters:
            mask = torch.zeros(max_seq, dtype=torch.bool)
            mask[: cl.size(0)] = True
            enc_masks.append(mask)
        enc_masks += [torch.zeros(max_seq, dtype=torch.bool)] * (len(padded_data) - len(clusters))
        encoder_mask = torch.stack(enc_masks).unsqueeze(0).to(device)

        enc_out = target_encoder(batch_tensor, encoder_mask)  # [1,P,S,D]
        enc_np = enc_out.squeeze(0).cpu().numpy()

        for p, (cl, ids) in enumerate(zip(clusters, cluster_face_ids)):
            real_len = cl.size(0)
            for j in range(real_len):
                fid = int(ids[j].item())
                face_features[fid] = enc_np[p, j, :]

    # Build colors aligned to all faces: compute for present ones, fill defaults for missing
    present_face_ids = [idx for idx, feat in enumerate(face_features) if feat is not None]
    face_features_present = np.array([feat for feat in face_features if feat is not None])
    if face_features_present.shape[0] != num_faces:
        missing_count = num_faces - face_features_present.shape[0]
        print(f"Warning: collected {face_features_present.shape[0]} features, expected {num_faces}. Missing {missing_count} faces.")
    if face_features_present.shape[0] == 0:
        rgba_full = np.tile(np.array([[180, 180, 180, 255]], dtype=np.uint8), (num_faces, 1))
    else:
        rgba_present = _pca_to_rgb(face_features_present)
        default_rgba = np.array([180, 180, 180, 255], dtype=np.uint8)
        rgba_full = np.tile(default_rgba[None, :], (num_faces, 1))
        for fid, color in zip(present_face_ids, rgba_present):
            rgba_full[fid] = color
    rgba = _propagate_colors_to_duplicate_faces(mesh, rgba_full)
    print('rgba.shape', rgba.shape)

    # Set per-face colors
    mesh.visual.face_colors = rgba

    # Derive per-vertex colors by averaging colors of incident faces
    num_vertices = mesh.vertices.shape[0]
    vertex_accum = np.zeros((num_vertices, 4), dtype=np.float64)
    vertex_counts = np.zeros((num_vertices, 1), dtype=np.int64)

    faces = mesh.faces
    for face_index, (v0, v1, v2) in enumerate(faces):
        color_rgba = rgba[face_index].astype(np.float64)
        vertex_accum[v0] += color_rgba
        vertex_accum[v1] += color_rgba
        vertex_accum[v2] += color_rgba
        vertex_counts[v0] += 1
        vertex_counts[v1] += 1
        vertex_counts[v2] += 1

    # Avoid division by zero; only average where counts > 0
    nonzero = vertex_counts.squeeze(-1) > 0
    averaged = np.zeros_like(vertex_accum)
    averaged[nonzero] = vertex_accum[nonzero] / vertex_counts[nonzero]
    v_rgba = averaged.clip(0, 255).astype(np.uint8)

    # Set per-vertex colors
    mesh.visual.vertex_colors = v_rgba
    os.makedirs(save_root, exist_ok=True)
    base = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(save_root, f"{base}_pca_colored.ply")
    if save_ascii:
        vertices_xyz = mesh.vertices.astype(np.float64)
        faces_idx = mesh.faces.astype(np.int64)
        vertex_colors_rgb = v_rgba[:, :3]
        face_colors_rgb = rgba[:, :3]
        _write_ascii_ply_with_colors(out_path, vertices_xyz, faces_idx, vertex_colors_rgb, face_colors_rgb)
    else:
        mesh.export(out_path)
    print(f"Saved PCA-colored mesh to {out_path}{' (ASCII custom PLY writer)' if save_ascii else ''}")

    # If classifier is available, also save classification labels
    if classifier is not None and num_classes is not None:
        print("Generating classification predictions...")
        
        # Initialize face and vertex labels
        face_labels = np.zeros(num_faces, dtype=np.int32)
        vertex_labels = np.zeros(num_vertices, dtype=np.int32)
        
        # Process each cluster to get classification predictions
        for idx in range(len(vis_ds)):
            clusters, cluster_face_indices, cluster_face_ids = vis_ds[idx]
            max_seq = max(cl.size(0) for cl in clusters)
            padded_data = [F.pad(cl, (0, 0, 0, max_seq - cl.size(0))) for cl in clusters]
            batch_tensor = torch.stack(padded_data).unsqueeze(0).to(device)

            enc_masks = []
            for cl in clusters:
                mask = torch.zeros(max_seq, dtype=torch.bool)
                mask[: cl.size(0)] = True
                enc_masks.append(mask)
            enc_masks += [torch.zeros(max_seq, dtype=torch.bool)] * (len(padded_data) - len(clusters))
            encoder_mask = torch.stack(enc_masks).unsqueeze(0).to(device)

            # Get encoder features
            enc_out = target_encoder(batch_tensor, encoder_mask)  # [1,P,S,D]
            
            # Get classification predictions
            logits = classifier(batch_tensor, encoder_mask)  # [1,P,S,num_classes]
            predictions = torch.argmax(logits, dim=-1)  # [1,P,S]
            pred_np = predictions.squeeze(0).cpu().numpy()

            # Assign predictions to faces
            for p, (cl, ids) in enumerate(zip(clusters, cluster_face_ids)):
                real_len = cl.size(0)
                for j in range(real_len):
                    fid = int(ids[j].item())
                    face_labels[fid] = pred_np[p, j]

        # Propagate face labels to duplicate faces
        face_labels = _propagate_labels_to_duplicate_faces(mesh, face_labels)

        # Derive vertex labels by majority voting from incident faces
        for face_index, (v0, v1, v2) in enumerate(mesh.faces):
            face_label = face_labels[face_index]
            # Simple assignment - vertices get the label of their first incident face
            # This could be improved with proper majority voting if needed
            if vertex_labels[v0] == 0:  # Only set if not already set
                vertex_labels[v0] = face_label
            if vertex_labels[v1] == 0:
                vertex_labels[v1] = face_label
            if vertex_labels[v2] == 0:
                vertex_labels[v2] = face_label

        # Fill any remaining unlabeled vertices with 0
        vertex_labels[vertex_labels == 0] = 0

        # Save mesh with classification labels
        if save_ascii:
            vertices_xyz = mesh.vertices.astype(np.float64)
            faces_idx = mesh.faces.astype(np.int64)
            label_out_path = os.path.join(save_root, f"{base}_classified.ply")
            _write_ascii_ply_with_labels(label_out_path, vertices_xyz, faces_idx, vertex_labels, face_labels)
            print(f"Saved classification-labeled mesh to {label_out_path}")
            
            # Print classification statistics
            unique_face_labels, face_counts = np.unique(face_labels, return_counts=True)
            unique_vertex_labels, vertex_counts = np.unique(vertex_labels, return_counts=True)
            print(f"Face label distribution: {dict(zip(unique_face_labels, face_counts))}")
            print(f"Vertex label distribution: {dict(zip(unique_vertex_labels, vertex_counts))}")
        else:
            # For binary PLY, we'd need to modify the mesh object directly
            print("Classification labels only supported with ASCII PLY format")


def main():
    parser = argparse.ArgumentParser(description="Visualize PCA-colored mesh features from a trained target encoder.")
    parser.add_argument("--config", required=True, type=str, help="Path to the training YAML config used to define the model and data.")
    # Optional overrides; if omitted, values will be read from config['visualization'] or other config sections
    parser.add_argument("--weights", required=False, type=str, help="Path to checkpoint .pth containing 'target_encoder'. Overrides config if given.")
    parser.add_argument("--save_root", required=False, type=str, default=None, help="Directory to save visualization outputs. Overrides config if given.")
    parser.add_argument("--device", required=False, type=str, choices=["cpu", "cuda"], help="Device to run on. Overrides config if given.")
    parser.add_argument("--n_clusters", type=int, default=None, help="Optional override for fixed number of clusters during visualization.")
    parser.add_argument("--mesh_dir", type=str, default=None, help="Optional override for mesh directory; overrides config if given.")
    parser.add_argument("--mesh_path", type=str, default=None, help="Optional single mesh file path to visualize.")
    parser.add_argument("--ascii", action="store_true", help="Save the output PLY in ASCII format instead of binary.")
    parser.add_argument("--legend", action="store_true", help="Also save a PCA->RGB legend image to visualize the color mapping.")
    parser.add_argument("--legend_pc3", type=float, default=None, help="If set, save a single-panel legend with PC3 (blue) fixed at this value in [0,1].")
    parser.add_argument("--legend_cube", action="store_true", help="Save a cube-style legend showing multiple faces of the 3D mapping.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Pull visualization defaults from config, with sensible fallbacks
    viz_cfg = config.get("visualization", {})

    # Resolve paths and options with precedence: CLI > visualization section > other config defaults
    mesh_dir = args.mesh_dir or viz_cfg.get("mesh_dir") or config.get("paths", {}).get("mesh_dir")
    single_mesh_path = args.mesh_path or viz_cfg.get("mesh_path")
    # If a single mesh is specified, prefer its directory to avoid requiring mesh_dir
    if single_mesh_path:
        mesh_dir = os.path.dirname(single_mesh_path)
    checkpoint_path = args.weights or viz_cfg.get("weights") or config.get("paths", {}).get("checkpoint_path")
    if checkpoint_path is None:
        # As a final fallback, try typical filenames inside checkpoint_dir
        ckpt_dir = config.get("paths", {}).get("checkpoint_dir")
        if ckpt_dir:
            candidates = [
                os.path.join(ckpt_dir, "best", "best_model.pth"),
                os.path.join(ckpt_dir, "best_model.pth"),
                os.path.join(ckpt_dir, "last_model.pth"),
            ]
            checkpoint_path = next((p for p in candidates if os.path.exists(p)), None)
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided. Specify --weights or set visualization.weights or paths.checkpoint_path in the config.")

    # model params
    feature_dim = config["model"]["feature_dim"]
    embedding_dim = config["model"]["embedding_dim"]
    num_heads = config["model"]["num_heads"]
    num_attention_blocks = config["model"]["num_attention_blocks"]
    dropout = config["model"].get("dropout", 0.1)
    faces_per_cluster = config["model"]["faces_per_cluster"]
    clusters_per_batch = config["model"]["clusters_per_batch"]
    pe = config["model"]["use_pe"]
    use_hierarchical = config["model"].get("use_hierarchical", False)
    n_clusters = args.n_clusters if args.n_clusters is not None else viz_cfg.get("n_clusters", config["model"]["n_clusters"])

    device_str = args.device or viz_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Build target encoder model and load weights
    # target_encoder = nomeformer(feature_dim, embedding_dim, num_heads, num_attention_blocks, dropout).to(device)
    target_encoder = nomeformer(
    feature_dim=feature_dim,            # your input feature size
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_attention_blocks=num_attention_blocks,
    dropout=dropout,
    summary_mode='cls',       # or 'avg'
    use_hierarchical=use_hierarchical,    # or False
    num_hierarchical_stages=1,
    fourier=False,             # keep as you use now
    relative_positional_encoding=False  # <-- activate RPE
).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Try to load from target_encoder first, fallback to model_state_dict for supervised models
    classifier = None
    num_classes = None
    
    if "target_encoder" in ckpt:
        print("Loading from target_encoder...")
        missing, unexpected = target_encoder.load_state_dict(ckpt["target_encoder"], strict=False)
    elif "model_state_dict" in ckpt:
        print("Loading from supervised model (model_state_dict)...")
        # Create a mapping to strip 'encoder.' prefix from keys
        state_dict = ckpt["model_state_dict"]
        encoder_state_dict = {}
        classifier_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                # Remove 'encoder.' prefix
                new_key = key[8:]  # Remove 'encoder.' (8 characters)
                encoder_state_dict[new_key] = value
            elif key.startswith("classifier."):
                # Extract classifier weights
                classifier_key = key[11:]  # Remove 'classifier.' (11 characters)
                classifier_state_dict[classifier_key] = value
        
        missing, unexpected = target_encoder.load_state_dict(encoder_state_dict, strict=False)
        
        # If we have classifier weights, build and load the classifier
        if classifier_state_dict:
            # Try to infer num_classes from the classifier weights
            if "weight" in classifier_state_dict:
                num_classes = classifier_state_dict["weight"].shape[0]
                print(f"Detected {num_classes} classes from classifier weights")
                
                # Build classifier
                classifier = DownstreamClassifier(
                    encoder=target_encoder,
                    num_classes=num_classes,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    shallow=True,  # Use shallow classifier by default
                    num_unfrozen_encoder_layers=0  # Keep encoder frozen
                ).to(device)
                
                # Load classifier weights into the classifier head submodule
                classifier_missing, classifier_unexpected = classifier.classifier.load_state_dict(classifier_state_dict, strict=False)
                if classifier_missing or classifier_unexpected:
                    print(f"Classifier loaded with missing keys: {classifier_missing} and unexpected keys: {classifier_unexpected}")
                classifier.eval()
                print("Successfully loaded classifier for classification visualization")
            else:
                print("Classifier weights found but could not determine number of classes")
    else:
        print("Loading from direct state_dict...")
        missing, unexpected = target_encoder.load_state_dict(ckpt, strict=False)
    
    if missing or unexpected:
        print(f"Loaded with missing keys: {missing} and unexpected keys: {unexpected}")
    target_encoder.eval()

    # Output directory
    if args.save_root is None:
        save_root = viz_cfg.get("save_root")
        if save_root is None:
            base_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            save_root = os.path.join(base_dir, "pca_visuals_from_ckpt")
    else:
        save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    # Optionally save legend image that explains the PCA->RGB color mapping
    if args.legend:
        if args.legend_pc3 is not None:
            z = float(np.clip(args.legend_pc3, 0.0, 1.0))
            legend_path = os.path.join(save_root, f"pca_rgb_legend_pc3_{z:.2f}.png")
            save_pca_rgb_legend(legend_path, single_z=z)
        else:
            legend_path = os.path.join(save_root, "pca_rgb_legend.png")
            save_pca_rgb_legend(legend_path)
        print(f"Saved PCA->RGB legend to {legend_path}")

    if args.legend_cube:
        cube_path = os.path.join(save_root, "pca_rgb_cube_faces.png")
        save_pca_rgb_cube(cube_path)
        print(f"Saved PCA->RGB cube faces legend to {cube_path}")

    visualize_random_mesh_pca(
        target_encoder=target_encoder,
        mesh_dir=mesh_dir,
        save_root=save_root,
        device=device,
        faces_per_cluster=faces_per_cluster,
        clusters_per_batch=clusters_per_batch,
        use_pe=pe,
        n_clusters_fixed=n_clusters,
        mesh_path=single_mesh_path,
        save_ascii=True,
        classifier=classifier,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()


