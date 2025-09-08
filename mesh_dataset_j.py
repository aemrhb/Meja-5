import os
import torch
from torch.utils.data import Dataset
import trimesh
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import torch.nn.functional as F
import pickle   
import time
import tempfile     

# Helper functions for clustering and reordering

def compute_cluster_centroids(vertices, labels, n_clusters):
    centroids = np.zeros((n_clusters, 3))
    counts = np.zeros(n_clusters)
    for i, v in enumerate(vertices):
        cluster_id = labels[i]
        centroids[cluster_id] += v
        counts[cluster_id] += 1
    centroids /= np.maximum(counts[:, None], 1)
    return centroids

def reorder_clusters_by_proximity(centroids):
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    mst = minimum_spanning_tree(dist_matrix).toarray()
    mst[mst == 0] = np.inf
    visited = set()
    ordered_clusters = []
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        ordered_clusters.append(node)
        for neighbor in np.argsort(mst[node]):
            if neighbor not in visited and np.isfinite(mst[node, neighbor]):
                dfs(neighbor)
    for start in range(len(centroids)):
        if start not in visited:
            dfs(start)
    if len(ordered_clusters) < len(centroids):
        remaining_clusters = list(set(range(len(centroids))) - set(ordered_clusters))
        remaining_clusters = sorted(
            remaining_clusters,
            key=lambda x: np.min(dist_matrix[x, ordered_clusters])
        )
        ordered_clusters.extend(remaining_clusters)
    return ordered_clusters

# MeshDataset with Dual Masking Strategy
class MeshDataset(Dataset):
    def __init__(self, mesh_dir, n_clusters, clusters_per_batch, PE, 
                 augmentation=None, transform=None, masking_ratio=0.3):
        """
        Args:
            mesh_dir (str): Directory with mesh files (.obj or .ply).
            n_clusters (int): Total number of clusters (for K-Means).
            clusters_per_batch (int): Number of clusters (patches) per sample.
            PE (bool): Whether to include positional encoding.
            augmentation (callable, optional): Mesh augmentation function.
            transform (callable, optional): Additional transform for features.
            masking_ratio (float): Fraction of clusters to mask (for target prediction).
        """
        self.mesh_dir = mesh_dir
        self.mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj') or f.endswith('.ply')]
        self.n_clusters = n_clusters
        self.clusters_per_batch = clusters_per_batch
        self.PE = PE
        self.augmentation = augmentation
        self.transform = transform
        self.masking_ratio = masking_ratio

        print(f"Found {len(self.mesh_files)} mesh files")

    def __len__(self):
        return len(self.mesh_files) * (self.n_clusters // self.clusters_per_batch)

    def __getitem__(self, idx):
        # Determine which mesh file and which batch of clusters (patches) to process
        mesh_idx = idx // (self.n_clusters // self.clusters_per_batch)
        cluster_batch_idx = idx % (self.n_clusters // self.clusters_per_batch)

        # Load the mesh
        mesh_file = self.mesh_files[mesh_idx]
        mesh_path = os.path.join(self.mesh_dir, mesh_file)
        mesh = trimesh.load(mesh_path, force='mesh')

        # Apply optional augmentation with some probability
        if self.augmentation and np.random.rand() < 0.1:
            mesh = self.augmentation(mesh)

        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Optionally, normalize the vertices to [0,1]
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)

        # K-Means clustering on vertices
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        vertex_labels = kmeans.fit_predict(vertices)

        # Compute centroids and reorder clusters based on spatial proximity
        centroids = compute_cluster_centroids(vertices, vertex_labels, self.n_clusters)
        ordered_cluster_indices = reorder_clusters_by_proximity(centroids)

        # Group faces into clusters based on the dominant vertex cluster of each face
        clusters = [[] for _ in range(self.n_clusters)]
        for face in faces:
            face_cluster_labels = vertex_labels[face]
            most_common_cluster = np.bincount(face_cluster_labels).argmax()
            clusters[most_common_cluster].append(face)

        # Reorder clusters according to our computed order
        clusters = [clusters[i] for i in ordered_cluster_indices]

        # Get the clusters for this batch
        start_cluster = cluster_batch_idx * self.clusters_per_batch
        end_cluster = start_cluster + self.clusters_per_batch
        selected_clusters = clusters[start_cluster:end_cluster]

        # Calculate additional features for each face (e.g., normals)
        mesh_normals = mesh.face_normals
        mesh_area = mesh.area_faces
        mesh_angle = mesh.face_angles



        # Normalize mesh_angle to range [0, 1]
        min_angle = mesh_angle.min()
        max_angle = mesh_angle.max()
        mesh_angle = (mesh_angle - min_angle) / (max_angle - min_angle)

        # Normalize mesh_area to range [0, 1]
        min_area = mesh_area.min()
        max_area = mesh_area.max()
        mesh_area = (mesh_area - min_area) / (max_area - min_area)

        # Precompute normalized coordinates for all faces
        precomputed_coords = [normalized_vertices[face].flatten().tolist() for face in faces]

        # Prepare the output in the desired nested list format with additional features
        nested_list_faces = []
        face_to_index = {tuple(face): i for i, face in enumerate(faces)}

        for cluster in selected_clusters:
            cluster_list_faces = []
            for face in cluster:
                face_features = []
                face_idx = face_to_index[tuple(face)]
                if self.PE:
                    coords = precomputed_coords[face_idx]
                    face_features.append(coords)

                face_features.append(mesh_angle[face_idx])
                # print('mesh_angle[face_idx]',mesh_angle[face_idx])
                face_features.append(mesh_normals[face_idx])
                # print('mesh_normals[face_idx]',mesh_normals[face_idx])
                face_features = np.concatenate(face_features).tolist()
                face_features.append(mesh_area[face_idx])
                # print('mesh_area[face_idx]',mesh_area[face_idx])

                cluster_list_faces.append(face_features)
            nested_list_faces.append(torch.tensor(np.array(cluster_list_faces)))

        if self.transform:
            nested_list_faces = self.transform(nested_list_faces)


        return nested_list_faces




    
def custom_collate_fn(batch, masking_ratio_range=(0.3, 0.7), clusters_to_select=10):
    """
    Custom collate function for MeshDataset, ensuring that:
    - A subset of clusters is selected per sample.
    - A random percentage (30%-70%) of faces within each selected cluster is masked.
    
    Returns:
        padded_batches: Tensor of shape (batch_size, max_patch_size, max_sequence_length, feature_dim)
        encoder_mask: Tensor of shape (batch_size, max_patch_size, max_sequence_length)
                      (True means the token is visible; note that added tokens are masked out)
        prediction_mask: Tensor of shape (batch_size, max_patch_size, max_sequence_length)
                         (True means the token is a target for prediction)
        added_token_mask: Tensor of shape (batch_size, max_patch_size, max_sequence_length)
                          (True indicates that the token was added via padding)
    """
    
    # Determine maximum number of patches and maximum sequence length across the batch
    max_patch_size = max(len(item) for item in batch)
    max_sequence_length = max(cluster.size(0) for item in batch for cluster in item)
    
    padded_batches = []      # Holds padded face features per sample
    encoder_masks = []       # Encoder mask (context mask with padded tokens masked out)
    prediction_masks = []    # Prediction mask (target mask)
    added_token_masks = []   # Mask that flags added (padded) tokens
    
    for clusters in batch:
        num_clusters = len(clusters)
        
        # Limit the number of clusters to select for masking
        clusters_to_sample = min(clusters_to_select, num_clusters)
        selected_clusters = np.random.choice(num_clusters, clusters_to_sample, replace=False)
        
        # Initialize masks for each cluster
        # For context: start with all ones (visible) and then hide some tokens
        context_mask = [torch.ones(cluster.size(0), dtype=torch.bool) for cluster in clusters]
        # For target: start with all zeros (not predicted) and then mark masked tokens
        target_mask = [torch.zeros(cluster.size(0), dtype=torch.bool) for cluster in clusters]
        
        for cluster_idx in selected_clusters:
            cluster_faces = clusters[cluster_idx]
            num_faces = cluster_faces.size(0)

            # Step 1: Randomly choose tokens to hide in context (masking)
            mask_fraction = np.random.uniform(*masking_ratio_range)
            num_faces_to_mask = max(1, int(num_faces * mask_fraction))
            masked_indices = np.random.choice(num_faces, num_faces_to_mask, replace=False)
            context_mask[cluster_idx][masked_indices] = False
            cluster_faces[masked_indices] = torch.zeros_like(cluster_faces[masked_indices])  # zero out

            # Step 2: Build prediction target mask:
            # - Select 15% of masked (hidden) tokens
            # - Select 5% of unmasked (visible) tokens
            masked_set = set(masked_indices)
            unmasked_set = set(range(num_faces)) - masked_set

            num_masked_to_predict = max(1, int(len(masked_set) * 0.15))
            num_unmasked_to_predict = max(0, int(len(unmasked_set) * 0.05))

            # Randomly sample from masked and unmasked sets
            masked_pred_indices = np.random.choice(list(masked_set), num_masked_to_predict, replace=False)
            unmasked_pred_indices = (
                np.random.choice(list(unmasked_set), num_unmasked_to_predict, replace=False)
                if num_unmasked_to_predict > 0 else []
            )

            # Combine into final prediction mask
            final_pred_indices = np.concatenate([masked_pred_indices, unmasked_pred_indices])
            target_mask[cluster_idx][final_pred_indices] = True
       
        # Pad each cluster's face features to the maximum sequence length
        padded_data = [
            torch.nn.functional.pad(cluster, (0, 0, 0, max_sequence_length - cluster.size(0)))
            for cluster in clusters
        ]
        # For samples with fewer clusters than max_patch_size, add completely padded clusters
        pad_cluster_tensor = torch.zeros(max_sequence_length, clusters[0].size(1))
        padded_data += [pad_cluster_tensor] * (max_patch_size - len(clusters))
        padded_batches.append(torch.stack(padded_data))
        
        # Pad context mask for each cluster: pad with False (0) for positions that were added
        padded_context = [
            torch.nn.functional.pad(mask, (0, max_sequence_length - mask.size(0)))
            for mask in context_mask
        ]
        padded_context += [torch.zeros(max_sequence_length, dtype=torch.bool)] * (max_patch_size - len(context_mask))
        encoder_masks.append(torch.stack(padded_context))
        
        # Pad target mask similarly, padding with False (0)
        padded_target = [
            torch.nn.functional.pad(mask, (0, max_sequence_length - mask.size(0)))
            for mask in target_mask
        ]
        padded_target += [torch.zeros(max_sequence_length, dtype=torch.bool)] * (max_patch_size - len(target_mask))
        prediction_masks.append(torch.stack(padded_target))
        
        # Build the added token mask:
        # For each cluster, mark padded positions as True and real tokens as False.
        added_masks = []
        for cluster in clusters:
            real_length = cluster.size(0)
            pad_length = max_sequence_length - real_length
            mask = torch.cat([
                torch.zeros(real_length, dtype=torch.bool), 
                torch.ones(pad_length, dtype=torch.bool)
            ])
            added_masks.append(mask)
        # For clusters that are added (to reach max_patch_size), all positions are padded (True)
        added_masks += [torch.ones(max_sequence_length, dtype=torch.bool)] * (max_patch_size - len(clusters))
        added_token_masks.append(torch.stack(added_masks))
    
    return (
        torch.stack(padded_batches),      # Padded patches
        torch.stack(encoder_masks),       # Encoder mask (context mask with padded tokens masked)
        torch.stack(prediction_masks),    # Prediction mask
        torch.stack(added_token_masks)    # Added token mask (indicates which tokens were padded)
    )
