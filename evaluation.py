import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from ignite.metrics import IoU, ConfusionMatrix
from torchmetrics import F1Score, Accuracy

from mesh_dataset_2 import MeshDataset, custom_collate_fn
from model_G_2 import nomeformer
from tools.downst import DownstreamClassifier
import torch


def build_model_from_config(config, device):
    feature_dim = config['model']['feature_dim']
    embedding_dim = config['model']['embedding_dim']
    num_heads = config['model']['num_heads']
    num_attention_blocks = config['model']['num_attention_blocks']
    N_class = config['model']['n_classes']
    dropout = config['model'].get('dropout', 0.1)
    use_hierarchical = config['model'].get('use_hierarchical', False)
    fourier = config['model'].get('fourier', False)
    relative_positional_encoding = config['model'].get('relative_positional_encoding', False)

    encoder = nomeformer(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        dropout=dropout,
        summary_mode='cls',
        use_hierarchical=use_hierarchical,
        num_hierarchical_stages=1,
        fourier=fourier,
        relative_positional_encoding=relative_positional_encoding,
    )
    model = DownstreamClassifier(encoder, N_class, embedding_dim, dropout, True, 0).to(device)
    return model


def evaluate(model, data_loader, num_classes, ignore_index, device):
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes)
    if ignore_index is not None:
        miou_metric = IoU(cm=cm, ignore_index=ignore_index)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none', ignore_index=ignore_index).to(device)
        acc_metric = Accuracy(task='multiclass', num_classes=num_classes, average='none', ignore_index=ignore_index).to(device)
    else:
        miou_metric = IoU(cm=cm)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none').to(device)
        acc_metric = Accuracy(task='multiclass', num_classes=num_classes, average='none').to(device)

    f1_metric.reset()
    acc_metric.reset()

    with torch.no_grad():
        for batch, labels, masks in data_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(batch, masks)
            pred = logits.reshape(-1, num_classes)
            target = labels.reshape(-1)
            mask = masks.view(-1)
            valid_mask = mask == 1
            pred = pred[valid_mask]
            target = target[valid_mask]
            cm.update((pred, target))
            f1_metric.update(pred, target)
            acc_metric.update(pred, target)

    f1_scores = f1_metric.compute()
    accuracy = acc_metric.compute()
    if ignore_index is not None and ignore_index < num_classes:
        class_mask = torch.arange(num_classes, device=f1_scores.device) != ignore_index
        mean_f1 = f1_scores[class_mask].mean().item()
        mean_acc = accuracy[class_mask].mean().item()
    else:
        mean_f1 = f1_scores.mean().item()
        mean_acc = accuracy.mean().item()
    miou = miou_metric.compute()

    return mean_f1, mean_acc, miou, f1_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained downstream model on a test directory.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config containing evaluation paths')
    parser.add_argument('--batch_size', type=int, default=None, help='Optional override for batch size')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_clusters = config['model']['n_clusters']
    clusters_per_batch = config['training']['clusters_per_batch']
    PE = config['model']['use_pe']
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    ignore_index = config['training'].get('ignore_index', None)
    num_classes = config['model']['n_classes']
    # Read evaluation paths from config
    test_mesh_dir = config['paths']['test_mesh_dir']
    test_label_dir = config['paths']['test_label_dir']
    test_json_dir = config['paths'].get('test_json_dir', None)
    checkpoint_path = config['paths']['checkpoint_path']

    dataset = MeshDataset(
        mesh_dir=test_mesh_dir,
        label_dir=test_label_dir,
        n_clusters=n_clusters,
        clusters_per_batch=clusters_per_batch,
        PE=PE,
        json_dir=test_json_dir,
        augmentation=None,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model = build_model_from_config(config, device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    current_state = model.state_dict()
    checkpoint_state = ckpt['model_state_dict']
    matched = {k: v for k, v in checkpoint_state.items() if k in current_state and current_state[k].shape == v.shape}
    current_state.update(matched)
    model.load_state_dict(current_state, strict=False)
    # Report classifier restoration explicitly
    classifier_keys = [k for k in matched.keys() if k.startswith('classifier.')]
    print(f"Restored {len(classifier_keys)} classifier params; matched {len(matched)} / {len(current_state)} total params.")
    model = model.to(device)

    mean_f1, mean_acc, miou, f1_scores = evaluate(model, data_loader, num_classes, ignore_index, device)

    print(f"Evaluation Results:\n  Mean F1: {mean_f1:.4f}\n  Mean Accuracy: {mean_acc:.4f}\n  mIoU: {miou}")
    print(f"Per-class F1: {f1_scores}")


if __name__ == '__main__':
    main()


