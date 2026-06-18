"""Run validation-session LOSO for MoE1 metadata-head + two_head + HMM.

For each validation session in a domain:
- train on original train rows plus all other validation sessions,
- validate/evaluate on the held-out validation session,
- fit the usual two_head combiner from frozen expert score exports,
- evaluate raw two_head and HMM(strength=8, mix=1, alpha=1).

Outputs are written under a new experiment root and do not overwrite existing
MoE1 experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"
EXPERIMENT_ROOT = MOE_ROOT / "experiments"
DATA_ROOT = MOE_ROOT / "moe_data"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}

import sys
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))
import ablate_moe1_hmm_decoding as hmm  # noqa: E402


@dataclass(frozen=True)
class Job:
    domain: str
    feature: str
    heldout_session: str
    fold_root: Path
    manifest: Path

    @property
    def run_name(self) -> str:
        return f"{self.domain.lower()}_{self.feature}_dyadic_tcn_k11_seed13"

    @property
    def run_dir(self) -> Path:
        return self.fold_root / "experts" / self.run_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MoE1 validation-session LOSO pipeline.")
    p.add_argument("--domains", nargs="+", choices=("CR", "CC", "cr", "cc"), default=["CR", "CC"])
    p.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "moe1_validation_loso_metadata_head_two_head_hmm")
    p.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--min-epochs", type=int, default=24)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--combiner-step", type=float, default=0.05)
    p.add_argument("--hmm-strength", type=float, default=8.0)
    p.add_argument("--hmm-mix", type=float, default=1.0)
    p.add_argument("--hmm-alpha", type=float, default=1.0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def base_manifest(feature: str, domain: str) -> Path:
    return DATA_ROOT / "outputs" / "windows_w2400_s1200_by_domain" / feature / f"{feature}_w2400_s1200_dyadic_{domain.lower()}.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validation_sessions(domain: str) -> list[str]:
    rows = read_csv(base_manifest(FEATURES[0], domain))
    return sorted({row["session_id"] for row in rows if row["model_split"] == "val_internal"})


def make_fold_manifests(domain: str, heldout_session: str, fold_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    manifest_dir = fold_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for feature in FEATURES:
        src = base_manifest(feature, domain)
        rows = read_csv(src)
        rewritten: list[dict[str, str]] = []
        for row in rows:
            row = dict(row)
            if row["model_split"] == "train_internal":
                row["model_split"] = "train_internal"
            elif row["model_split"] == "val_internal":
                row["model_split"] = "val_internal" if row["session_id"] == heldout_session else "train_internal"
            else:
                # We do not need test exports for LOSO; keeping test rows out of
                # test_internal prevents large per-fold test_prediction_scores files.
                row["model_split"] = "unused_test_internal"
            rewritten.append(row)
        dst = manifest_dir / f"{feature}_dyadic_{domain.lower()}_heldout_{heldout_session}.csv"
        write_csv(dst, rewritten)
        out[feature] = dst
    return out


def run_command(command: list[str], log_path: Path, gpu: str | None, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    line = " ".join(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} ===\n$ {line}\n")
        log.flush()
        if dry_run:
            return
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {line}")


def train_job(args: argparse.Namespace, job: Job, gpu: str) -> None:
    complete = job.run_dir / ".complete_loso"
    if complete.exists():
        print(f"skip_complete {job.domain} heldout={job.heldout_session} {job.feature}", flush=True)
        return
    cmd = [
        str(args.python), str(MOE_ROOT / "train_moe1_metadata_head_tcn.py"),
        "--manifest", str(job.manifest),
        "--metadata", str(DATA_ROOT / "outputs" / "participant_metadata.csv"),
        "--metadata-mode", "age_gender",
        "--metadata-dropout", "0.2",
        "--output-root", str(job.fold_root / "experts"),
        "--run-name", job.run_name,
        "--levels", "5",
        "--kernel-size", "11",
        "--hidden-channels", "64",
        "--dropout", "0.2",
        "--causal-tcn",
        "--train-split", "train_internal",
        "--val-split", "val_internal",
        "--test-split", "test_internal",
        "--mmap-cache-root", str(DATA_ROOT / "processed" / "domain_norm_mmap"),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--seed", str(args.seed),
        "--device", "cuda",
        "--resume",
    ]
    run_command(cmd, job.fold_root / "logs" / f"train_{job.run_name}.log", gpu, args.dry_run)
    eval_cmd = [
        str(args.python), str(MOE_ROOT / "evaluate_moe1_metadata_head_checkpoint.py"),
        "--run-dir", str(job.run_dir),
        "--split", "train_internal",
        "--checkpoint", "model_best.pt",
        "--output-dir", str(job.run_dir / "diagnostics" / "train_internal"),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", "cuda",
    ]
    run_command(eval_cmd, job.fold_root / "logs" / f"eval_train_{job.run_name}.log", gpu, args.dry_run)
    if not args.dry_run:
        complete.write_text("complete\n", encoding="utf-8")


def train_fold_experts(args: argparse.Namespace, domain: str, heldout_session: str, fold_root: Path, manifests: dict[str, Path]) -> None:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    jobs = queue.Queue()
    for feature in FEATURES:
        jobs.put(Job(domain, feature, heldout_session, fold_root, manifests[feature]))
    errors: list[BaseException] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                train_job(args, job, gpu)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"Expert training failed for {domain} heldout={heldout_session}: {errors[0]}")


def fit_combiner(args: argparse.Namespace, domain: str, fold_root: Path) -> None:
    out = fold_root / "combiners"
    summary = out / "two_head" / "summary.json"
    if summary.exists():
        print(f"skip_combiner {domain} {fold_root.name}", flush=True)
        return
    cmd = [
        str(args.python), str(MOE_ROOT / "fit_moe1_combiner.py"),
        "--domain", domain,
        "--expert-root", str(fold_root / "experts"),
        "--output-root", str(out),
        "--metadata", str(DATA_ROOT / "outputs" / "participant_metadata.csv"),
        "--step", str(args.combiner_step),
        "--modes", "two_head",
    ]
    run_command(cmd, fold_root / "logs" / "fit_combiner.log", None, args.dry_run)


def add_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return hmm.add_mean_rows(rows)


def evaluate_fold_hmm(args: argparse.Namespace, domain: str, heldout_session: str, fold_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ns = SimpleNamespace(
        cc_expert_root=fold_root / "experts" if domain == "CC" else None,
        cr_expert_root=fold_root / "experts" if domain == "CR" else None,
        cc_combiner_root=fold_root / "combiners" if domain == "CC" else None,
        cr_combiner_root=fold_root / "combiners" if domain == "CR" else None,
        transition_alpha=args.hmm_alpha,
        mix=args.hmm_mix,
    )
    train_keys, train_labels, _ = hmm.read_domain_data(domain, "train", ns)
    val_keys, val_labels, val_logits_by_feature = hmm.read_domain_data(domain, "val", ns)
    weights = hmm.read_weights(domain, ns)
    combined_logits = hmm.combine_two_head(val_keys, val_logits_by_feature, weights)
    base_pred = combined_logits.argmax(axis=1)
    matrices = hmm.transition_matrices(train_keys, train_labels, args.hmm_alpha, args.hmm_mix)
    log_probs = hmm.log_softmax_by_head(val_keys, combined_logits)
    hmm_pred = hmm.apply_hmm(val_keys, log_probs, matrices, args.hmm_strength)
    raw_rows = []
    class_rows = []
    for mode, param, pred in [
        ("two_head_raw", "none", base_pred),
        ("two_head_hmm", f"mix={args.hmm_mix:g};strength={args.hmm_strength:g};alpha={args.hmm_alpha:g}", hmm_pred),
    ]:
        rows = hmm.evaluate(val_keys, val_labels, pred, domain, mode, param)
        for row in rows:
            row["heldout_session"] = heldout_session
        raw_rows.extend(rows)
        cls = hmm.class_metric_rows(val_keys, val_labels, pred, domain, mode, param)
        for row in cls:
            row["heldout_session"] = heldout_session
        class_rows.extend(cls)
    metric_rows = add_mean_rows(raw_rows)
    write_csv(fold_root / "hmm_results.csv", metric_rows)
    write_csv(fold_root / "class_metrics.csv", class_rows)
    return metric_rows, class_rows


def aggregate_domain(output_root: Path, domain: str, all_rows: list[dict[str, object]], class_rows: list[dict[str, object]]) -> None:
    write_csv(output_root / f"{domain.lower()}_fold_hmm_results.csv", all_rows)
    write_csv(output_root / f"{domain.lower()}_fold_class_metrics.csv", class_rows)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in all_rows:
        if row.get("head") not in {"task", "social", "mean"}:
            continue
        key = (str(row["mode"]), str(row["param"]), str(row["head"]))
        grouped.setdefault(key, []).append(float(row["kappa"]))
    summary = [
        {"domain": domain, "mode": mode, "param": param, "head": head, "n_folds": len(vals), "mean_kappa": float(np.mean(vals)), "std_kappa": float(np.std(vals))}
        for (mode, param, head), vals in sorted(grouped.items())
    ]
    write_csv(output_root / f"{domain.lower()}_summary.csv", summary)


def combined_summary(output_root: Path) -> None:
    rows = []
    for domain in ("CR", "CC"):
        p = output_root / f"{domain.lower()}_summary.csv"
        if p.exists() and p.stat().st_size:
            with p.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["mode"], row["param"], row["head"])
        grouped.setdefault(key, []).append(float(row["mean_kappa"]))
    combined = [
        {"mode": mode, "param": param, "head": head, "n_domains": len(vals), "domain_mean_kappa": float(np.mean(vals))}
        for (mode, param, head), vals in sorted(grouped.items())
        if len(vals) == 2
    ]
    write_csv(output_root / "combined_summary.csv", combined)


def main() -> None:
    args = parse_args()
    args.domains = [d.upper() for d in args.domains]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    for domain in args.domains:
        domain_rows: list[dict[str, object]] = []
        domain_class_rows: list[dict[str, object]] = []
        sessions = validation_sessions(domain)
        print(f"domain={domain} validation_sessions={sessions}", flush=True)
        for heldout in sessions:
            fold_root = args.output_root / domain.lower() / f"heldout_{heldout}"
            fold_root.mkdir(parents=True, exist_ok=True)
            print(f"start_fold domain={domain} heldout={heldout} root={fold_root}", flush=True)
            manifests = make_fold_manifests(domain, heldout, fold_root)
            train_fold_experts(args, domain, heldout, fold_root, manifests)
            fit_combiner(args, domain, fold_root)
            if args.dry_run:
                print(f"dry_run_complete_fold domain={domain} heldout={heldout}", flush=True)
                continue
            metric_rows, class_rows = evaluate_fold_hmm(args, domain, heldout, fold_root)
            domain_rows.extend(metric_rows)
            domain_class_rows.extend(class_rows)
            aggregate_domain(args.output_root, domain, domain_rows, domain_class_rows)
            combined_summary(args.output_root)
            print(f"complete_fold domain={domain} heldout={heldout}", flush=True)
    combined_summary(args.output_root)
    print(f"complete_loso output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
