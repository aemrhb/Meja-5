import os
import pickle
import numpy as np
import trimesh
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import time
import tempfile


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
    visited, ordered = set(), []
    def dfs(u):
        if u in visited: return
        visited.add(u); ordered.append(u)
        for v in np.argsort(mst[u]):
            if v not in visited and np.isfinite(mst[u, v]):
                dfs(v)
    for i in range(len(centroids)):
        dfs(i)
    # any left-over clusters?
    if len(ordered) < len(centroids):
        remaining = set(range(len(centroids))) - set(ordered)
        remaining = sorted(remaining, key=lambda x: np.min(dist_matrix[x, ordered]))
        ordered.extend(remaining)
    return ordered

class MeshDataset(Dataset):
    def __init__(self, mesh_dir, clusters_per_batch, faces_per_cluster=200,
                 PE=True, augmentation=None, transform=None, 
                 flexible_num_clusters=False, n_clusters=None):
        self.mesh_dir = mesh_dir
        self.clusters_per_batch = clusters_per_batch
        self.faces_per_cluster = faces_per_cluster
        self.PE = PE
        self.augmentation = augmentation
        self.transform = transform
        self.flexible_num_clusters = flexible_num_clusters
        self.n_clusters_fixed = n_clusters

        if not self.flexible_num_clusters:
            if self.n_clusters_fixed is None or self.n_clusters_fixed <= 0:
                raise ValueError("When flexible_num_clusters is False, a positive n_clusters must be provided.")

        # where to store per-mesh cluster caches
        self.cache_dir = os.path.join(mesh_dir, ".cluster_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._in_memory_cache = {}  # mesh_path -> (clusters, face_normals, face_angles, face_areas, precomputed_coords)

        # find all mesh files
        self.mesh_files = [
            f for f in os.listdir(mesh_dir)
            if f.endswith(('.obj', '.ply', '.off'))]
        

        # build index map: one entry per “batch” that getitem will produce
        self.index_map = []
        for fname in self.mesh_files:
            path = os.path.join(mesh_dir, fname)
            mesh = trimesh.load(path, force='mesh')
            n_faces = mesh.faces.shape[0]
            if self.flexible_num_clusters:
                n_clusters = max(1, n_faces // faces_per_cluster)
            else:
                n_clusters = int(self.n_clusters_fixed)
            n_batches = max(1, n_clusters // clusters_per_batch)
            for b in range(n_batches):
                self.index_map.append((path, n_clusters, b))
        print(f"→ built index_map with {len(self.index_map)} total samples")

    def __len__(self):
        return len(self.index_map)


    def _compute_and_cache(self, mesh_path, n_clusters):
        # 1) load mesh
        t0 = time.perf_counter()
        mesh = trimesh.load(mesh_path, force='mesh')
        verts, faces = mesh.vertices, mesh.faces
    
        # 2) compute face-centroids
        face_centroids = verts[faces].mean(axis=1)
    
        # 3) run KMeans on those centroids
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        flabels = kmeans.fit_predict(face_centroids)
    
        # 4) group faces by label
        raw_clusters = [[] for _ in range(n_clusters)]
        for face_idx, lbl in enumerate(flabels):
            raw_clusters[lbl].append(faces[face_idx])
    
        # 5) compute centroids for each cluster to reorder them
        centroids = compute_cluster_centroids(face_centroids, flabels, n_clusters)
        order = reorder_clusters_by_proximity(centroids)
    
        # 6) build `clusters` in the final, reordered order *and* drop any empties
        clusters = [raw_clusters[i] for i in order if len(raw_clusters[i]) > 0]
    
        # 7) build face→index map once and for all
        face_to_index = {tuple(face): i for i, face in enumerate(faces)}
    
        # 8) precompute all per-face features
        normals = mesh.face_normals
        angles  = mesh.face_angles
        areas   = mesh.area_faces

        # Normalize angles and areas to [0, 1] for consistency with mesh_dataset.py
        a_min, a_max = angles.min(), angles.max()
        a_den = (a_max - a_min) if (a_max - a_min) != 0 else 1e-8
        angles = (angles - a_min) / a_den

        ar_min, ar_max = areas.min(), areas.max()
        ar_den = (ar_max - ar_min) if (ar_max - ar_min) != 0 else 1e-8
        areas = (areas - ar_min) / ar_den
    
        # 9) normalized vertex coords per face (for PE)
        min_c, max_c = verts.min(0), verts.max(0)
        norm_verts = (verts - min_c) / (max_c - min_c)
        precomp_coords = [norm_verts[f].flatten() for f in faces]
    
        # 10) write atomically to disk and store in memory
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{n_clusters}.pkl"
        )
        payload = {
            'clusters': clusters,
            'face_normals': normals,
            'face_angles': angles,
            'face_areas': areas,
            'precomp_coords': precomp_coords,
            'face_to_index': face_to_index
        }
        
        # 10a) dump into a temp file in the same directory
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)
        with os.fdopen(fd, 'wb') as tmpf:
            pickle.dump(payload, tmpf)
        
        # 10b) atomically rename to the real cache filename
        os.replace(tmp_path, cache_file)
        
        # 10c) store in memory
        self._in_memory_cache[mesh_path] = payload
        
        dt = time.perf_counter() - t0
        print(f"[CACHE] computed & saved clusters for {os.path.basename(mesh_path)} "
              f"in {dt:.2f}s")

    def __getitem__(self, idx):
        mesh_path, n_clusters, batch_idx = self.index_map[idx]
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{n_clusters}.pkl"
        )

        # CASE A: not in in‐memory dict
        if mesh_path not in self._in_memory_cache:
            # A1: disk cache exists?
            if os.path.exists(cache_file):
                t0 = time.perf_counter()
                print(f"[CACHE LOAD] disk hit for {os.path.basename(mesh_path)}")
                with open(cache_file, 'rb') as f:
                    payload = pickle.load(f)
                dt = time.perf_counter() - t0
                print(f"[CACHE LOAD] loaded in {dt:.3f}s")
                self._in_memory_cache[mesh_path] = payload
            else:
                print(f"[CACHE MISS] no cache for {os.path.basename(mesh_path)}, computing…")
                self._compute_and_cache(mesh_path, n_clusters)
        else:
            # CASE B: already in memory
            print(f"[CACHE MEM] using in‐memory cache for {os.path.basename(mesh_path)}")

        cache = self._in_memory_cache[mesh_path]

        clusters        = cache['clusters']
        face_normals    = cache['face_normals']
        face_angles     = cache['face_angles']
        face_areas      = cache['face_areas']
        precomp_coords  = cache['precomp_coords']
        face_to_index   = cache['face_to_index']

        # select your slice of clusters
        selected = clusters[batch_idx * self.clusters_per_batch:
                             (batch_idx+1) * self.clusters_per_batch]

        nested_list_faces = []
        nested_list_face_indices = []  # vertex triplets for adjacency/masking
        nested_list_face_ids = []      # original face indices for mapping back
        for cl in selected:
            feats = []
            indices_triplets = []
            indices_ids = []
            for face in cl:
                # look up the original face-index in O(1) instead of searching:
                fid = face_to_index[tuple(face)]
                face_feats = []
                if self.PE:
                    face_feats.append(precomp_coords[fid])
                face_feats.append(face_angles[fid])
                face_feats.append(face_normals[fid])
                face_feats = np.concatenate(face_feats).tolist()
                face_feats.append(face_areas[fid])
                feats.append(face_feats)
                # keep both: vertex triplets (for masking) and original face id (for mapping)
                indices_triplets.append(face)
                indices_ids.append(fid)
            nested_list_faces.append(torch.tensor(feats))
            nested_list_face_indices.append(torch.tensor(indices_triplets))
            nested_list_face_ids.append(torch.tensor(indices_ids, dtype=torch.long))

        if self.transform:
            nested_list_faces = self.transform(nested_list_faces)
        # return triple for richer downstream use; maintain order
        return nested_list_faces, nested_list_face_indices, nested_list_face_ids




    
def custom_collate_fn(batch, masking_ratio_range=(0.3, 0.7), clusters_to_select=10, num_pred_blocks=5):
    """
    Custom collate function for MeshDataset, ensuring that:
    - A subset of clusters is selected per sample.
    - A random percentage (30%-70%) of faces within each selected cluster is masked (contiguously).
    - Multiple prediction mask blocks are generated per sample (num_pred_blocks), each predicting a random subset of the masked faces.
    Returns:
        padded_batches: Tensor of shape (batch_size, max_patch_size, max_sequence_length, feature_dim)
        encoder_mask: Tensor of shape (batch_size, max_patch_size, max_sequence_length)
                      (True means the token is visible; note that added tokens are masked out)
        prediction_masks_blocks: Tensor of shape (batch_size, num_pred_blocks, max_patch_size, max_sequence_length)
                         (True means the token is a target for prediction)
        added_token_mask: Tensor of shape (batch_size, max_patch_size, max_sequence_length)
                          (True indicates that the token was added via padding)
    """
    
    # Determine maximum number of patches and maximum sequence length across the batch
    max_patch_size = max(len(item[0]) for item in batch)
    max_sequence_length = max(max(patch.size(0) for patch in item[0]) for item in batch)

    padded_batches = []      # Holds padded face features per sample
    encoder_masks = []       # Encoder mask (context mask with padded tokens masked out)
    prediction_masks_blocks = []    # List of prediction mask blocks per sample
    added_token_masks = []   # Mask that flags added (padded) tokens

    for clusters, clusters_face_indices, clusters_face_ids in batch:
        num_clusters = len(clusters)
        clusters_to_sample = min(clusters_to_select, num_clusters)
        selected_clusters = np.random.choice(num_clusters, clusters_to_sample, replace=False)

        # --- Compute context mask and masked indices for each cluster (ONCE) ---
        context_mask = []  # List of torch.bool tensors, one per cluster
        masked_indices_per_cluster = []  # List of lists, one per cluster
        for cluster_idx, (cluster, cluster_face_indices) in enumerate(zip(clusters, clusters_face_indices)):
            num_faces = cluster.size(0)
            if cluster_idx in selected_clusters:
                cluster_face_indices_np = cluster_face_indices.numpy()  # shape [F, 3] vertex ids
                mask_fraction = np.random.uniform(*masking_ratio_range)
                num_faces_to_mask = max(1, int(num_faces * mask_fraction))
                face_neighbors = get_adjacent_faces(cluster_face_indices_np)
                masked_indices = contiguous_mask(face_neighbors, num_faces_to_mask)
                mask = torch.ones(num_faces, dtype=torch.bool)
                mask[masked_indices] = False  # Masked faces are 0 (not visible)
                context_mask.append(mask)
                masked_indices_per_cluster.append(masked_indices)
            else:
                mask = torch.ones(num_faces, dtype=torch.bool)
                context_mask.append(mask)
                masked_indices_per_cluster.append([])

        # --- For each prediction block, generate a separate prediction mask ---
        pred_blocks = []
        for _ in range(num_pred_blocks):
            target_mask = [torch.zeros(cluster.size(0), dtype=torch.bool) for cluster in clusters]
            for cluster_idx in selected_clusters:
                masked_indices = masked_indices_per_cluster[cluster_idx]
                if len(masked_indices) == 0:
                    continue
                pred_fraction = np.random.uniform(0.2, 0.5)
                num_masked_to_predict = max(1, int(len(masked_indices) * pred_fraction))
                masked_pred_indices = np.random.choice(masked_indices, num_masked_to_predict, replace=False)
                target_mask[cluster_idx][masked_pred_indices] = True
            # Pad target mask for each cluster
            padded_target = [
                torch.nn.functional.pad(mask, (0, max_sequence_length - mask.size(0)))
                for mask in target_mask
            ]
            padded_target += [torch.zeros(max_sequence_length, dtype=torch.bool)] * (max_patch_size - len(target_mask))
            pred_blocks.append(torch.stack(padded_target))
        # Stack all prediction blocks for this sample
        prediction_masks_blocks.append(torch.stack(pred_blocks))  # shape: (num_pred_blocks, max_patch_size, max_sequence_length)

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
        torch.stack(padded_batches),      # (batch_size, max_patch_size, max_sequence_length, feature_dim)
        torch.stack(encoder_masks),       # (batch_size, max_patch_size, max_sequence_length)
        torch.stack(prediction_masks_blocks),  # (batch_size, num_pred_blocks, max_patch_size, max_sequence_length)
        torch.stack(added_token_masks)    # (batch_size, max_patch_size, max_sequence_length)
    )

def get_adjacent_faces(faces):
    # faces: (N, 3) array of vertex indices
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for idx, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1)%3])))
            edge_to_faces[edge].append(idx)
    face_neighbors = [[] for _ in range(len(faces))]
    for edge, face_idxs in edge_to_faces.items():
        if len(face_idxs) > 1:
            for f in face_idxs:
                face_neighbors[f].extend([fi for fi in face_idxs if fi != f])
    # Remove duplicates
    face_neighbors = [list(set(nbs)) for nbs in face_neighbors]
    return face_neighbors

def contiguous_mask(face_neighbors, num_to_mask):
    import random
    N = len(face_neighbors)
    seed = random.randint(0, N-1)
    mask = set([seed])
    frontier = set(face_neighbors[seed])
    while len(mask) < num_to_mask and frontier:
        next_face = frontier.pop()
        if next_face not in mask:
            mask.add(next_face)
            frontier.update(face_neighbors[next_face])
    # If not enough, fill randomly
    while len(mask) < num_to_mask:
        mask.add(random.randint(0, N-1))
    return list(mask)
