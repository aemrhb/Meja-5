import os
import json
import pickle
import numpy as np
import trimesh
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import tempfile
import time
import torch.nn.functional as F
import trimesh.transformations as tf
#from tools.helper import kmeans_with_size_cap
class MeshAugmentation:
    """
    Applies random augmentations to a mesh, such as rotation, scaling, noise, and flipping.
    Args:
        rotation_range (float): Maximum rotation angle in degrees.
        scale_range (tuple): Min and max scaling factors.
        noise_std (float): Standard deviation of Gaussian noise.
        flip_probability (float): Probability of flipping the mesh.
    """
    def __init__(self, rotation_range=15, scale_range=(0.9, 1.1), noise_std=0.01, flip_probability=0.5):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.flip_probability = flip_probability

    def __call__(self, mesh, mode='RSN'):
        """
        Applies augmentations to the mesh.
        Args:
            mesh (trimesh.Trimesh): The mesh to augment.
            mode (str): Which augmentations to apply ('R'=rotation, 'S'=scaling, 'N'=noise, 'F'=flip).
        Returns:
            mesh (trimesh.Trimesh): The augmented mesh.
        """
        # Apply flipping (if enabled and randomly selected)
        if 'F' in mode and np.random.rand() < 0.5:
            if np.random.rand() < self.flip_probability:
                mesh.invert()

        # Apply rotation (if enabled and randomly selected)
        if 'R' in mode and np.random.rand() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=angle, 
                direction=axis, 
                point=mesh.centroid
            )
            mesh.apply_transform(rotation_matrix)

        # Apply scaling (if enabled and randomly selected)
        if 'S' in mode and np.random.rand() < 0.5:
            # Create a scaling matrix
            scaling_matrix = tf.scale_matrix(self.scale_range)
            # Apply the scaling to the mesh
            mesh.apply_transform(scaling_matrix)

        # Apply noise (if enabled and randomly selected)
        if 'N' in mode and np.random.rand() < 0.5:
            noise = np.random.randn(*mesh.vertices.shape) * self.noise_std
            mesh.vertices += noise

        return mesh

# ---------- helpers: clustering like your first code ----------
def compute_cluster_centroids(vertices, labels, n_clusters):
    centroids = np.zeros((n_clusters, 3), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for i, v in enumerate(vertices):
        c = labels[i]
        centroids[c] += v
        counts[c] += 1
    counts = np.maximum(counts, 1)  # avoid divide-by-zero
    centroids /= counts[:, None]
    return centroids

def reorder_clusters_by_proximity(centroids):
    if len(centroids) <= 1:
        return list(range(len(centroids)))
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
    if len(ordered) < len(centroids):
        remaining = set(range(len(centroids))) - set(ordered)
        remaining = sorted(remaining, key=lambda x: np.min(dist_matrix[x, ordered]))
        ordered.extend(remaining)
    return ordered


class MeshDataset(Dataset):
    """
    Mesh dataset that:
      • clusters faces via KMeans on face *centroids*
      • uses faces_per_cluster -> dynamic n_clusters
      • reorders clusters by spatial proximity
      • caches per-mesh payloads (atomic write + in-memory)
      • supports classification (one label per mesh) or segmentation (per-face labels)
      • optional positional encoding (PE) of per-face vertex coords
      • optional JSON side-features loading (returned alongside sample if you want)
    """
    def __init__(
        self,
        mesh_dir,
        label_dir,
        clusters_per_batch,
        faces_per_cluster=200,
        PE=False,
        json_dir=None,
        augmentation=None,
        transform=None,
        classification=False,
        additional_geometrical_features=False
    ):
        self.mesh_dir = mesh_dir
        self.label_dir = label_dir
        self.json_dir = json_dir
        self.clusters_per_batch = clusters_per_batch
        self.faces_per_cluster = faces_per_cluster
        self.PE = PE
        self.augmentation = augmentation   # when not None, we compute on-the-fly (no disk cache)
        self.transform = transform
        self.is_classification = classification
        self.AGF = additional_geometrical_features

        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label dir not found: {label_dir}")
        if self.json_dir and not os.path.isdir(self.json_dir):
            raise FileNotFoundError(f"JSON dir not found: {self.json_dir}")

        self.mesh_files = [
            f for f in os.listdir(mesh_dir)
            if f.endswith(('.obj', '.ply', '.off'))
        ]
        print(f"Found {len(self.mesh_files)} mesh files")
        print(
            f"Dataset params: faces_per_cluster={self.faces_per_cluster}, "
            f"clusters_per_batch={self.clusters_per_batch}, PE={int(self.PE)}, AGF={int(self.AGF)}"
        )

        # caching
        self.cache_dir = os.path.join(mesh_dir, ".cluster_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._in_memory_cache = {}  # mesh_path -> payload

        # labels
        self.labels = self._load_labels()

        # build index map like the first code (batch slices of clusters)
        self.index_map = []
        for fname in self.mesh_files:
            path = os.path.join(mesh_dir, fname)
            mesh = trimesh.load(path, force='mesh')
            n_faces = mesh.faces.shape[0]
            import math
            n_clusters = max(1, math.ceil(n_faces / self.faces_per_cluster))

            if self.is_classification:
                self.index_map.append((path, n_clusters, None))
            else:
                n_batches = max(1, math.ceil(n_clusters / self.clusters_per_batch))
                for b in range(n_batches):
                    self.index_map.append((path, n_clusters, b))
        print(f"→ built index_map with {len(self.index_map)} total samples")

    # ---------- labels / json ----------
    def _load_labels(self):
        """
        Load .txt labels. For classification: one int per mesh.
        For segmentation: list of ints per face.
        """
        labels = {}
        for file_name in os.listdir(self.label_dir):
            if not file_name.endswith('.txt'):
                continue
            base = os.path.splitext(file_name)[0]
            with open(os.path.join(self.label_dir, file_name), 'r') as f:
                if self.is_classification:
                    labels[base] = int(f.read().strip())
                else:
                    labels[base] = [int(x) for x in f.read().split()]
        print(f"Loaded labels for {len(labels)} files.")
        return labels

    def _load_json_features(self, base_name):
        if not self.json_dir:
            return None
        p = os.path.join(self.json_dir, base_name + '.json')
        if os.path.isfile(p):
            with open(p, 'r') as f:
                return json.load(f)
        return None

    def __len__(self):
        return len(self.index_map)

    # ---------- cache compute (face-centroid KMeans, like your first code) ----------
    def _compute_and_cache(self, mesh_path, n_clusters, base_name):
        t0 = time.perf_counter()
        mesh = trimesh.load(mesh_path, force='mesh')
        verts, faces = mesh.vertices, mesh.faces

        # labels (broadcast for classification if needed)
        if base_name not in self.labels:
            raise KeyError(f"No labels for {base_name}")
        if self.is_classification:
            face_labels = [self.labels[base_name]] * len(faces)
        else:
            face_labels = self.labels[base_name]
            if len(face_labels) != len(faces):
                raise ValueError(
                    f"Face labels ({len(face_labels)}) != faces ({len(faces)}) for {base_name}"
                )

        # face centroids -> KMeans on centroids (not vertices)
        face_centroids = verts[faces].mean(axis=1)  # [F,3]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        flabels = kmeans.fit_predict(face_centroids)
    #    flabels = fps_voronoi_labels(face_centroids, n_clusters)
    # face_centroids = verts[faces].mean(axis=1)
        # flabels = kmeans_with_size_cap(
        #     mesh,
        #     n_clusters=n_clusters,
        #     faces_per_cluster=self.faces_per_cluster,
        #     prefer_adjacent=True,   # keep patches contiguous when possible
        #     verbose=False
        # )

        # group faces by cluster label
        raw_clusters = [[] for _ in range(n_clusters)]
        for face_idx, lbl in enumerate(flabels):
            raw_clusters[lbl].append(faces[face_idx])

        # reorder clusters by proximity of *cluster centroids*
        centroids = compute_cluster_centroids(face_centroids, flabels, n_clusters)
        order = reorder_clusters_by_proximity(centroids)
        clusters = [raw_clusters[i] for i in order if len(raw_clusters[i]) > 0]

        # map face tuple -> original index
        face_to_index = {tuple(face): i for i, face in enumerate(faces)}

        # per-face features from mesh
        normals = mesh.face_normals
        angles  = mesh.face_angles
        areas   = mesh.area_faces

        # normalize vertex coords to [0,1] then flatten per face for PE
        min_c, max_c = verts.min(0), verts.max(0)
        denom = np.where((max_c - min_c) == 0, 1.0, (max_c - min_c))
        norm_verts = (verts - min_c) / denom
        precomp_coords = [norm_verts[f].flatten() for f in faces]

        # additional geometrical features (per-face)
        add_slope = None
        add_height = None
        add_roughness = None
        if getattr(self, 'AGF', False):
            # slope/inclination: angle to +Z; use cos(theta)=|nz| -> theta=arccos(|nz|)
            nz = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
            slope = np.arccos(nz)  # radians, [0..pi/2]

            # height above ground: fit ground plane to lowest-z vertices and compute distances
            z_vals = verts[:, 2]
            z_thresh = np.percentile(z_vals, 5.0)
            ground_pts = verts[z_vals <= z_thresh]
            if ground_pts.shape[0] >= 3:
                gp_mean = ground_pts.mean(axis=0)
                gp_centered = ground_pts - gp_mean
                cov = gp_centered.T @ gp_centered / max(ground_pts.shape[0] - 1, 1)
                evals, evecs = np.linalg.eigh(cov)
                n = evecs[:, np.argmin(evals)]
                # ensure normal points upward
                if n[2] < 0:
                    n = -n
                n = n / (np.linalg.norm(n) + 1e-12)
                d = -np.dot(n, gp_mean)
                # signed distance (above ground should be positive given n points upward)
                norm_height = face_centroids @ n + d
            else:
                # fallback: use min-z reference
                z_min = z_vals.min()
                norm_height = face_centroids[:, 2] - z_min

            # local roughness: std of neighbor centroid heights within a radius
            # radius scaled by mesh extent
            bbox_diag = float(np.linalg.norm(max_c - min_c))
            radius = 0.05 * bbox_diag if bbox_diag > 0 else 0.05
            # naive neighborhood with pairwise distances (faces can be large; acceptable for moderate sizes)
            # compute pairwise distances in xy for locality; could also use full xyz
            from scipy.spatial.distance import cdist as _cdist
            dmat = _cdist(face_centroids, face_centroids)
            rough = np.zeros(len(face_centroids), dtype=np.float32)
            for i in range(len(face_centroids)):
                neigh = np.where(dmat[i] <= radius)[0]
                if neigh.size <= 1:
                    rough[i] = 0.0
                else:
                    rough[i] = float(np.std(norm_height[neigh]))

            add_slope = slope
            add_height = norm_height
            add_roughness = rough

        payload = {
            'clusters': clusters,                 # list[list[face(3)]]
            'face_normals': normals,              # [F,3]
            'face_angles': angles,                # [F,3]
            'face_areas': areas,                  # [F]
            'precomp_coords': precomp_coords,     # list[F][9]
            'face_to_index': face_to_index,
            'face_labels': face_labels,           # list[F] (or broadcasted single label)
            'add_slope': add_slope,
            'add_height': add_height,
            'add_roughness': add_roughness,
        }

        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{n_clusters}_PE{int(self.PE)}_AGF{int(self.AGF)}.pkl"
        )
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)
        with os.fdopen(fd, 'wb') as tmpf:
            pickle.dump(payload, tmpf)
        os.replace(tmp_path, cache_file)

        self._in_memory_cache[mesh_path] = payload
        print(
            f"[CACHE WRITE] {os.path.basename(mesh_path)} "
            f"(n_clusters={n_clusters}, faces_per_cluster={self.faces_per_cluster}, PE={int(self.PE)}, AGF={int(self.AGF)}) "
            f"in {time.perf_counter()-t0:.2f}s"
        )

    # ---------- getitem ----------
    def __getitem__(self, idx):
        mesh_path, n_clusters, batch_idx = self.index_map[idx]
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{n_clusters}_PE{int(self.PE)}_AGF{int(self.AGF)}.pkl"
        )

        use_cache = (self.augmentation is None)
        if use_cache:
            if mesh_path not in self._in_memory_cache:
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            payload = pickle.load(f)
                        if not isinstance(payload, dict):
                            raise ValueError("Invalid cache payload type")
                        self._in_memory_cache[mesh_path] = payload
                        print(
                            f"[CACHE LOAD] {os.path.basename(mesh_path)} "
                            f"(n_clusters={n_clusters}, faces_per_cluster={self.faces_per_cluster}, PE={int(self.PE)}, AGF={int(self.AGF)})"
                        )
                    except (EOFError, pickle.UnpicklingError, ValueError):
                        # recompute
                        print(
                            f"[CACHE INVALID] {os.path.basename(mesh_path)} — removing and recomputing "
                            f"(n_clusters={n_clusters}, faces_per_cluster={self.faces_per_cluster}, PE={int(self.PE)}, AGF={int(self.AGF)})"
                        )
                        try:
                            os.remove(cache_file)
                        except FileNotFoundError:
                            pass
                        self._compute_and_cache(mesh_path, n_clusters, base_name)
                else:
                    print(
                        f"[CACHE MISS] {os.path.basename(mesh_path)} — computing "
                        f"(n_clusters={n_clusters}, faces_per_cluster={self.faces_per_cluster}, PE={int(self.PE)}, AGF={int(self.AGF)})"
                    )
                    self._compute_and_cache(mesh_path, n_clusters, base_name)
        else:
            # on-the-fly compute (no disk write) — identical logic to _compute_and_cache but kept local
            mesh = trimesh.load(mesh_path, force='mesh')
            if self.augmentation and np.random.rand() < 0.2:
                mesh = self.augmentation(mesh)
            verts, faces = mesh.vertices, mesh.faces
            if self.is_classification:
                face_labels = [self.labels[base_name]] * len(faces)
            else:
                face_labels = self.labels[base_name]
                if len(face_labels) != len(faces):
                    raise ValueError(f"labels!=faces for {base_name}")
            face_centroids = verts[faces].mean(axis=1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            flabels = kmeans.fit_predict(face_centroids)
            raw_clusters = [[] for _ in range(n_clusters)]
            for face_idx, lbl in enumerate(flabels):
                raw_clusters[lbl].append(faces[face_idx])
            centroids = compute_cluster_centroids(face_centroids, flabels, n_clusters)
            order = reorder_clusters_by_proximity(centroids)
            clusters = [raw_clusters[i] for i in order if len(raw_clusters[i]) > 0]
            face_to_index = {tuple(face): i for i, face in enumerate(faces)}
            normals = mesh.face_normals
            angles  = mesh.face_angles
            areas   = mesh.area_faces
            min_c, max_c = verts.min(0), verts.max(0)
            denom = np.where((max_c - min_c) == 0, 1.0, (max_c - min_c))
            norm_verts = (verts - min_c) / denom
            precomp_coords = [norm_verts[f].flatten() for f in faces]
            # additional geometrical features
            add_slope = None
            add_height = None
            add_roughness = None
            if getattr(self, 'AGF', False):
                nz = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
                slope = np.arccos(nz)
                # height above ground via fitted plane on lowest 5% z vertices
                z_vals = verts[:, 2]
                z_thresh = np.percentile(z_vals, 5.0)
                ground_pts = verts[z_vals <= z_thresh]
                if ground_pts.shape[0] >= 3:
                    gp_mean = ground_pts.mean(axis=0)
                    gp_centered = ground_pts - gp_mean
                    cov = gp_centered.T @ gp_centered / max(ground_pts.shape[0] - 1, 1)
                    evals, evecs = np.linalg.eigh(cov)
                    n = evecs[:, np.argmin(evals)]
                    if n[2] < 0:
                        n = -n
                    n = n / (np.linalg.norm(n) + 1e-12)
                    d = -np.dot(n, gp_mean)
                    norm_height = face_centroids @ n + d
                else:
                    z_min = z_vals.min()
                    norm_height = face_centroids[:, 2] - z_min
                from scipy.spatial.distance import cdist as _cdist
                bbox_diag = float(np.linalg.norm(max_c - min_c))
                radius = 0.05 * bbox_diag if bbox_diag > 0 else 0.05
                dmat = _cdist(face_centroids, face_centroids)
                rough = np.zeros(len(face_centroids), dtype=np.float32)
                for i in range(len(face_centroids)):
                    neigh = np.where(dmat[i] <= radius)[0]
                    if neigh.size <= 1:
                        rough[i] = 0.0
                    else:
                        rough[i] = float(np.std(norm_height[neigh]))
                add_slope = slope
                add_height = norm_height
                add_roughness = rough

            payload = {
                'clusters': clusters,
                'face_normals': normals,
                'face_angles': angles,
                'face_areas': areas,
                'precomp_coords': precomp_coords,
                'face_to_index': face_to_index,
                'face_labels': face_labels,
                'add_slope': add_slope,
                'add_height': add_height,
                'add_roughness': add_roughness,
            }
            # keep only in-memory for this sample
            self._in_memory_cache[mesh_path] = payload

        cache = self._in_memory_cache[mesh_path]
        clusters        = cache['clusters']
        face_normals    = cache['face_normals']
        face_angles     = cache['face_angles']
        face_areas      = cache['face_areas']
        precomp_coords  = cache['precomp_coords']
        face_to_index   = cache['face_to_index']
        face_labels     = cache['face_labels']
        add_slope       = cache.get('add_slope', None)
        add_height      = cache.get('add_height', None)
        add_roughness   = cache.get('add_roughness', None)

        # pick cluster slice
        if self.is_classification:
            selected = clusters  # all clusters in one sample
        else:
            start = batch_idx * self.clusters_per_batch
            end   = start + self.clusters_per_batch
            selected = clusters[start:end]

        # build nested tensors (faces -> features) and labels if segmentation
        nested_list_faces = []
        nested_list_labels = []  # only used for segmentation

        for cl in selected:
            feats = []
            labs = []
            for face in cl:
                fid = face_to_index[tuple(face)]  # original face index
                fparts = []
                if self.PE:
                    fparts.append(precomp_coords[fid])  # 9-d coords
                fparts.append(face_angles[fid])         # 3
                fparts.append(face_normals[fid])        # 3
                if getattr(self, 'AGF', False) and add_slope is not None:
                    fparts.append([add_slope[fid]])     # 1
                    fparts.append([add_height[fid]])    # 1
                    fparts.append([add_roughness[fid]]) # 1
                fvec = np.concatenate(fparts).tolist()
                fvec.append(face_areas[fid])            # +1
                feats.append(fvec)

                if not self.is_classification:
                    labs.append(face_labels[fid])

            if feats:
                nested_list_faces.append(torch.tensor(np.array(feats), dtype=torch.float32))
                if not self.is_classification:
                    nested_list_labels.append(torch.tensor(np.array(labs), dtype=torch.long))

        if self.transform:
            nested_list_faces = self.transform(nested_list_faces)
            if not self.is_classification:
                nested_list_labels = self.transform(nested_list_labels)

        if self.is_classification:
            # return all clusters + single mesh label
            mesh_label = self.labels[base_name]
            # optionally also return json features if you want:
            # json_feats = self._load_json_features(base_name)
            return nested_list_faces, mesh_label
        else:
            # segmentation: return cluster slice + per-face labels aligned to feats
            return nested_list_faces, nested_list_labels



# Custom collate function to handle varying sizes of vertices and faces
def custom_collate_fn(batch):
    """
    Pads and stacks a batch of mesh data with varying sizes for use in a DataLoader.
    If is_classification=True in the dataset, each batch element contains all clusters for a mesh.
    Args:
        batch (list): List of tuples (nested_list_faces, nested_list_labels) from MeshDataset.__getitem__.
    Returns:
        padded_batches (Tensor): Batched and padded face features, shape (B, P, S, F).
        padded_labels (Tensor): Batched and padded labels, shape (B, P, L) or (B,) if classification.
        masks (Tensor): Mask tensor indicating valid entries, shape (B, P, S).
    """
    # Check if classification mode (label is int)
    if isinstance(batch[0][1], int):
        # Classification mode: labels are integers
        max_patch_size = max(len(item[0]) for item in batch)
        max_sequence_length = max(max(patch.size(0) for patch in item[0]) for item in batch)
        padded_batches = []
        masks = []
        for data, _ in batch:
            padded_data = [F.pad(patch, (0, 0, 0, max_sequence_length - patch.size(0))) for patch in data]
            padded_data += [torch.zeros(max_sequence_length, data[0].size(1))] * (max_patch_size - len(data))
            padded_batches.append(torch.stack(padded_data))
            batch_mask = torch.zeros((max_patch_size, max_sequence_length), dtype=torch.bool)
            for i, patch in enumerate(data):
                batch_mask[i, :patch.size(0)] = 1
            masks.append(batch_mask)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return torch.stack(padded_batches), labels, torch.stack(masks)
    # Original (segmentation) mode
    max_patch_size = max(len(item[0]) for item in batch)
    max_sequence_length = max(max(patch.size(0) for patch in item[0]) for item in batch)
    max_label_length = max(max(patch.size(0) for patch in item[1]) for item in batch)
    padded_batches = []
    padded_labels = []
    masks = []
    for data, label in batch:
        padded_data = [F.pad(patch, (0, 0, 0, max_sequence_length - patch.size(0))) for patch in data]
        padded_data += [torch.zeros(max_sequence_length, data[0].size(1))] * (max_patch_size - len(data))
        padded_batches.append(torch.stack(padded_data))
        padded_label = [F.pad(patch, (0, max_label_length - patch.size(0))) for patch in label]
        padded_label += [torch.zeros(max_label_length)] * (max_patch_size - len(label))
        padded_labels.append(torch.stack(padded_label))
        batch_mask = torch.zeros((max_patch_size, max_sequence_length), dtype=torch.bool)
        for i, patch in enumerate(data):
            batch_mask[i, :patch.size(0)] = 1
        masks.append(batch_mask)
    return torch.stack(padded_batches), torch.stack(padded_labels), torch.stack(masks)

