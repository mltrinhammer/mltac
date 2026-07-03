#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path('/workspace/ACM/ACM-clean')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MOE_ROOT = PROJECT_ROOT / 'MoE'
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))
THIS_DIR = MOE_ROOT / 'pinsoro_noxi_settings'
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from MoE.pinsoro_noxi_settings import apply_person_interaction_hmm_active_heads as hmm
from MoE.pinsoro_noxi_settings.train_gated_fusion import filter_domain, modality_dims, reconstruct
from MoE.pinsoro_noxi_settings.train_person_interaction_fusion_temporal_hidden_partner import (
    RoleMetadataPinSoRoWindowDataset,
    SharedPersonFusionInteractionTCN,
    metadata_dim_for_mode,
)
from src.acm_pipeline.pinsoro import read_class_labels
from src.acm_pipeline.pinsoro_data import PinSoRoWindowDataset, read_pinsoro_window_manifests
from src.acm_pipeline.pinsoro_train_utils import write_prediction_scores
from train_pinsoro_tcn import make_loader

HEAD_TO_STEM = {'task': 'task_engagement', 'social': 'social_engagement'}
RUNS = [
    {
        'name': 'cc_task_shared_linear',
        'head': 'task',
        'run_dir': 'MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13',
    },
    {
        'name': 'cc_social_shared_hidden_attention',
        'head': 'social',
        'run_dir': 'MoE/experiments/pinsoro_cc_social_shared_partner_ablation_delta010_metadata/pinsoro_cc_social_shared_hidden_attention_4p8s_shared_tcn_delta010_metadata_seed13',
    },
]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_metadata_table(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    table = {}
    if not path.exists():
        return table
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            table[(row['source_split'], row['session_id'], row['role'])] = row
    return table


def export_full_val_scores(run_dir: Path, output_score: Path, batch_size: int, num_workers: int) -> None:
    if output_score.exists() and output_score.stat().st_size > 0:
        return
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    manifests = [resolve(item) for item in config['manifest']]
    val_split = config.get('val_split', 'val_internal')
    domain_scope = config.get('domain_scope', 'CC')
    val_windows = filter_domain(read_pinsoro_window_manifests(manifests, PROJECT_ROOT, val_split), domain_scope)
    if not val_windows:
        raise RuntimeError(f'No validation windows for {run_dir}')
    metadata_mode = config.get('metadata_mode', 'none')
    mmap_cache_root = config.get('mmap_cache_root')
    mmap_cache_root = None if mmap_cache_root is None else resolve(mmap_cache_root)
    if metadata_mode != 'none':
        metadata_path = resolve(config['metadata']) if config.get('metadata') else MOE_ROOT / 'moe_data' / 'outputs' / 'participant_metadata.csv'
        metadata_table = load_metadata_table(metadata_path)
        dataset = RoleMetadataPinSoRoWindowDataset(
            val_windows,
            max_cached_tensors=2,
            mmap_cache_root=mmap_cache_root,
            project_root=PROJECT_ROOT,
            metadata_mode=metadata_mode,
            metadata_table=metadata_table,
            age_mean=float(config.get('metadata_age_mean', 0.0)),
            age_std=float(config.get('metadata_age_std', 1.0)),
        )
    else:
        dataset = PinSoRoWindowDataset(val_windows, 2, mmap_cache_root, PROJECT_ROOT)
    args = argparse.Namespace(batch_size=batch_size, num_workers=num_workers, seed=int(config.get('seed', 13)))
    loader = make_loader(dataset, args, shuffle=False, pin_memory=False)
    dims = {str(k): int(v) for k, v in config['modality_dims'].items()}
    model = SharedPersonFusionInteractionTCN(
        modality_dims=dims,
        fusion_mode=config.get('fusion_mode', 'concat'),
        fusion_channels=int(config.get('fusion_channels', 64)),
        person_hidden_channels=int(config.get('person_hidden_channels', 64)),
        person_levels=int(config.get('person_levels', 5)),
        person_kernel_size=int(config.get('person_kernel_size', 11)),
        dropout=float(config.get('dropout', 0.2)),
        modality_dropout=float(config.get('modality_dropout', 0.1)),
        causal_tcn=bool(config.get('causal_tcn', True)),
        encoder_sharing=config.get('encoder_sharing', 'shared'),
        interaction_mode=config.get('interaction_mode', 'none'),
        interaction_hidden_channels=int(config.get('interaction_hidden_channels', 32)),
        interaction_kernel_size=int(config.get('interaction_kernel_size', 5)),
        interaction_scale=float(config.get('interaction_scale', 0.0)),
        hidden_interaction_mode=config.get('hidden_interaction_mode', 'none'),
        hidden_interaction_scale=float(config.get('hidden_interaction_scale', 0.0)),
        hidden_interaction_window_steps=int(config.get('hidden_interaction_window_steps', 1)),
        hidden_attention_full=bool(config.get('hidden_attention_full', False)),
        head_architecture=config.get('head_architecture', 'shared_tcn'),
        head_adapter_levels=int(config.get('head_adapter_levels', 1)),
        metadata_dim=metadata_dim_for_mode(metadata_mode),
        metadata_embedding_dim=int(config.get('metadata_embedding_dim', 16)),
        metadata_dropout=float(config.get('metadata_dropout', 0.2)),
    )
    ckpt = torch.load(run_dir / 'model_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    reconstructed = reconstruct(model, dataset, loader, torch.device('cpu'))
    output_score.parent.mkdir(parents=True, exist_ok=True)
    write_prediction_scores(output_score, reconstructed, supervised_only=False)


def read_scores(path: Path, head: str):
    keys=[]; logits=[]
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['domain'] != 'CC' or row['head'] != head:
                continue
            n = hmm.CLASS_COUNTS[head]
            key = (row['domain'], row['source_split'], row['session_id'], row['role'], head, int(row['frame_idx']))
            arr = np.full(hmm.MAX_CLASSES, -1e9, dtype=np.float64)
            arr[:n] = [float(row[f'logit_{i}']) for i in range(n)]
            keys.append(key); logits.append(arr)
    return keys, np.stack(logits)


def cached_labels(cache_root: Path, key: hmm.Key, annotator: int | None):
    _domain, source_split, session_id, role, head, _frame = key
    stem = HEAD_TO_STEM[head]
    fname = f'{role}.{stem}.annotation.csv' if annotator is None else f'{role}.{stem}.{annotator}.annotation.csv'
    path = cache_root / source_split / session_id / fname
    if not path.exists():
        return None
    return read_class_labels(path, stem)


def build_targets(keys: list[hmm.Key], cache_root: Path, annotators: list[int]):
    cache = {}
    out = []
    coverage = {'canonical_valid': 0, 'blank_with_numbered': 0, 'blank_without_numbered': 0, 'n_total': len(keys)}
    for key in keys:
        head = key[4]; n = hmm.CLASS_COUNTS[head]; frame = key[5]
        ckey = (key[1], key[2], key[3], head, None)
        if ckey not in cache:
            cache[ckey] = cached_labels(cache_root, key, None)
        canonical = cache[ckey]
        dist = np.zeros(n, dtype=np.float64)
        source = 'missing'
        confidence = 0.0
        if canonical is not None:
            labels, mask = canonical
            if frame < len(labels) and frame < len(mask) and mask[frame] > 0 and 0 <= int(labels[frame]) < n:
                dist[int(labels[frame])] = 1.0
                source = 'canonical'
                confidence = 1.0
                coverage['canonical_valid'] += 1
        if source != 'canonical':
            votes = []
            for annotator in annotators:
                akey = (key[1], key[2], key[3], head, annotator)
                if akey not in cache:
                    cache[akey] = cached_labels(cache_root, key, annotator)
                loaded = cache[akey]
                if loaded is None:
                    continue
                labels, mask = loaded
                if frame < len(labels) and frame < len(mask) and mask[frame] > 0 and 0 <= int(labels[frame]) < n:
                    votes.append(int(labels[frame]))
            if votes:
                for vote in votes:
                    dist[vote] += 1.0
                dist /= float(len(votes))
                confidence = float(dist.max())
                source = 'numbered_blank'
                coverage['blank_with_numbered'] += 1
            else:
                coverage['blank_without_numbered'] += 1
        out.append({'dist': dist, 'source': source, 'confidence': confidence})
    return out, coverage


def soft_kappa(conf: np.ndarray) -> float:
    n = float(conf.sum())
    if n <= 0:
        return float('nan')
    obs = float(np.trace(conf) / n)
    exp = float(conf.sum(axis=1) @ conf.sum(axis=0) / (n*n))
    return float((obs-exp)/(1-exp)) if exp < 1 else float('nan')


def score(keys, logits, pred, targets, head, mode, param, run_name):
    n = hmm.CLASS_COUNTS[head]
    rows=[]
    for subset in ('canonical', 'numbered_blank', 'augmented'):
        conf = np.zeros((n,n), dtype=np.float64)
        confw = np.zeros((n,n), dtype=np.float64)
        nll = nllw = wsum = 0.0
        frames = 0
        for i, target in enumerate(targets):
            if subset == 'canonical' and target['source'] != 'canonical':
                continue
            if subset == 'numbered_blank' and target['source'] != 'numbered_blank':
                continue
            if subset == 'augmented' and target['source'] not in {'canonical','numbered_blank'}:
                continue
            ypred = int(pred[i])
            if not 0 <= ypred < n:
                continue
            dist = target['dist']
            if dist.sum() <= 0:
                continue
            probs = np.exp(logits[i,:n] - logits[i,:n].max()); probs /= probs.sum()
            w = float(target['confidence']) if target['source'] == 'numbered_blank' else 1.0
            conf[:, ypred] += dist
            confw[:, ypred] += w * dist
            nll_i = float(-(dist * np.log(np.clip(probs, 1e-12, 1.0))).sum())
            nll += nll_i; nllw += w*nll_i; wsum += w; frames += 1
        rows.append({
            'run': run_name, 'head': head, 'mode': mode, 'param': param, 'target_subset': subset,
            'n_frames': frames, 'soft_kappa': soft_kappa(conf), 'confidence_weighted_soft_kappa': soft_kappa(confw),
            'soft_nll': nll/frames if frames else float('nan'),
            'confidence_weighted_soft_nll': nllw/wsum if wsum else float('nan'),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=MOE_ROOT / 'experiments' / 'cc_annotator_sensitivity_2906')
    ap.add_argument('--cache-root', type=Path, default=PROJECT_ROOT / 'cache' / 'pinsoro')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--annotators', nargs='+', type=int, default=[1,2,3])
    ap.add_argument('--transition-strengths', nargs='+', type=float, default=[0.25,0.5,1.0,2.0,4.0,8.0,12.0])
    ap.add_argument('--transition-mixes', nargs='+', type=float, default=[0.5,0.75,1.0])
    ap.add_argument('--transition-alpha', type=float, default=1.0)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows=[]; coverage_rows=[]
    for spec in RUNS:
        run_dir = resolve(spec['run_dir'])
        head = spec['head']
        out_score = args.output_root / 'full_val_scores' / spec['name'] / 'val_prediction_scores.csv.gz'
        print(f'export_full {spec["name"]}', flush=True)
        export_full_val_scores(run_dir, out_score, args.batch_size, args.num_workers)
        keys, logits = read_scores(out_score, head)
        targets, cov = build_targets(keys, args.cache_root, args.annotators)
        coverage_rows.append({'run': spec['name'], 'head': head, **cov})
        raw_pred = logits.argmax(axis=1)
        all_rows.extend(score(keys, logits, raw_pred, targets, head, 'raw', 'none', spec['name']))
        manifests = [resolve(p) for p in json.loads((run_dir / 'config.json').read_text())['manifest']]
        train_keys, train_labels = hmm.read_train_labels(manifests, 'CC', 'train_internal')
        log_probs = hmm.log_softmax_by_head(keys, logits)
        for mix in args.transition_mixes:
            matrices = hmm.transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
            for strength in args.transition_strengths:
                pred = hmm.apply_hmm(keys, log_probs, matrices, strength)
                param = f'mix={mix};strength={strength};alpha={args.transition_alpha}'
                all_rows.extend(score(keys, logits, pred, targets, head, 'hmm', param, spec['name']))
    write_csv(args.output_root / 'sensitivity_metrics.csv', all_rows)
    write_csv(args.output_root / 'coverage.csv', coverage_rows)
    # Selected settings by canonical and augmented target subsets.
    selected=[]
    for run in sorted({r['run'] for r in all_rows}):
        for subset in ('canonical','numbered_blank','augmented'):
            rows=[r for r in all_rows if r['run']==run and r['target_subset']==subset]
            rows=[r for r in rows if not math.isnan(float(r['confidence_weighted_soft_kappa']))]
            if not rows:
                continue
            best=max(rows, key=lambda r: float(r['confidence_weighted_soft_kappa']))
            selected.append(best)
    write_csv(args.output_root / 'selected_by_confidence_weighted_soft_kappa.csv', selected)
    print(args.output_root, flush=True)

if __name__ == '__main__':
    main()
