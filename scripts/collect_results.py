"""Collect experiment results and format a comparison table.

Scans ACM/outputs/experiments/ for completed runs, reads their metric CSVs,
and prints a markdown table comparable to the MultiMediate26 organizer baseline.

All metrics are on the validation set (test labels held by organizers).

Usage:
    python ACM/scripts/collect_results.py
    python ACM/scripts/collect_results.py --experiments-dir ACM/outputs/experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model type display order for unimodal turn sweeps.
TURN_MODEL_ORDER = [
    "turns_simple_tcn",
    "turns_dyadic_shared",
    "turns_attention",
]
MODEL_ORDER = TURN_MODEL_ORDER

MODEL_LABELS = {
    "turns_simple_tcn": "Turn-Level Simple TCN",
    "turns_dyadic_shared": "Turn-Level Dyadic (shared head)",
    "turns_attention": "Turn-Level Attention TCN",
}

TRAINER_ORDER = ["simple", "dyadic_shared", "attention"]
TRAINER_LABELS = {
    "simple": "Simple TCN",
    "dyadic_shared": "Dyadic (shared head)",
    "attention": "Attention TCN",
}

MODEL_TO_TRAINER = {
    "turns_simple_tcn": "simple",
    "turns_dyadic_shared": "dyadic_shared",
    "turns_attention": "attention",
}
TRAINER_TO_TURN_MODEL = {trainer: model for model, trainer in MODEL_TO_TRAINER.items()}

# Feature set display order (voice first, then text, then video — matching organizer).
FEATURE_ORDER = [
    "audio_egemaps",
    "audio_w2vbert2",
    "text_xlm_roberta",
    "visual_openface",
    "visual_openpose",
    "visual_clip",
    "visual_dino",
    "visual_swin",
    "visual_videomae",
]

FEATURE_LABELS = {
    "audio_egemaps": "eGeMAPS v2",
    "audio_w2vbert2": "w2vBERT2",
    "text_xlm_roberta": "XLM-RoBERTa",
    "visual_openface": "OpenFace 2+3",
    "visual_openpose": "OpenPose",
    "visual_clip": "CLIP",
    "visual_dino": "DINOv2",
    "visual_swin": "SwinTransformer",
    "visual_videomae": "VideoMAE",
}

FEATURE_FAMILIES = {
    "audio_egemaps": "audio",
    "audio_w2vbert2": "audio",
    "text_xlm_roberta": "text",
    "visual_openface": "visual",
    "visual_openpose": "visual",
    "visual_clip": "visual",
    "visual_dino": "visual",
    "visual_swin": "visual",
    "visual_videomae": "visual",
}

FAMILY_ORDER = ["audio", "text", "visual"]
FAMILY_LABELS = {
    "audio": "Audio",
    "text": "Text",
    "visual": "Visual",
}

# Organizer MLP baseline CCC (combined across test sets) for reference.
ORGANIZER_BASELINE = {
    "audio_egemaps": 0.4529,
    "audio_w2vbert2": 0.2222,
    "text_xlm_roberta": 0.0793,
    "visual_openface": 0.1433,
    "visual_openpose": 0.0505,
    "visual_clip": 0.1474,
    "visual_dino": 0.1285,
    "visual_swin": 0.1463,
    "visual_videomae": 0.0955,
}

PERMUTATION_SEED = 13
DEFAULT_SIGNIFICANCE_ALPHA = 0.05
DEFAULT_PERMUTATION_SAMPLES = 100000
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_EXACT_LIMIT = 20


def read_metric_csv(path: Path) -> dict[str, str]:
    """Read a single-row metric CSV and return its fields."""
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_role_metrics(path: Path) -> dict[str, float]:
    """Read metrics_by_role.csv and return CCC per role."""
    if not path.exists():
        return {}
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            role = row.get("role", "")
            ccc = row.get("ccc", "")
            if role and ccc:
                try:
                    result[role] = float(ccc)
                except ValueError:
                    pass
    return result


def read_session_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Read metrics_by_session.csv keyed by (dataset, session_id)."""

    if not path.exists():
        return {}

    result: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            dataset = row.get("dataset", "")
            session_id = row.get("session_id", "")
            ccc_text = row.get("ccc", "")
            n_frames_text = row.get("n_frames", "")
            if not dataset or not session_id:
                continue
            try:
                ccc = float(ccc_text)
            except ValueError:
                ccc = float("nan")
            try:
                n_frames = float(n_frames_text)
            except ValueError:
                n_frames = float("nan")
            result[(dataset, session_id)] = {"ccc": ccc, "n_frames": n_frames}
    return result


def model_label(model_type: str) -> str:
    """Return a human-readable label for turn, window, or multimodal run types."""

    if model_type in MODEL_LABELS:
        return MODEL_LABELS[model_type]
    if model_type.startswith("windows_multimodal_"):
        suffix = model_type.removeprefix("windows_multimodal_")
        backbone, fusion = suffix.rsplit("_", 1)
        return f"Legacy Windows Multimodal {TRAINER_LABELS.get(backbone, backbone)} ({fusion})"
    if model_type.startswith("windows_"):
        backbone = model_type.removeprefix("windows_")
        return f"Legacy Windows {TRAINER_LABELS.get(backbone, backbone)}"
    if model_type.startswith("turns_multimodal_"):
        suffix = model_type.removeprefix("turns_multimodal_")
        backbone, fusion = suffix.rsplit("_", 1)
        return f"Multimodal {TRAINER_LABELS.get(backbone, backbone)} ({fusion})"
    return model_type


def combo_label(combo_name: str) -> str:
    """Pretty-print a multimodal combination name built from feature-set ids."""

    return " + ".join(FEATURE_LABELS.get(part, part) for part in combo_name.split("__"))


def multimodal_model_sort_key(model_type: str) -> tuple[int, int, str]:
    if model_type.startswith("turns_multimodal_"):
        suffix = model_type.removeprefix("turns_multimodal_")
    elif model_type.startswith("windows_multimodal_"):
        suffix = model_type.removeprefix("windows_multimodal_")
    else:
        suffix = model_type
    backbone, fusion = suffix.rsplit("_", 1)
    backbone_rank = TRAINER_ORDER.index(backbone) if backbone in TRAINER_ORDER else len(TRAINER_ORDER)
    fusion_rank = 0 if fusion == "concat" else 1
    return backbone_rank, fusion_rank, fusion


def fmt_delta(val: float, precision: int = 4) -> str:
    if val != val:
        return "-"
    return f"{val:+.{precision}f}"


def fmt_interval(low: float, high: float, precision: int = 4) -> str:
    if low != low or high != high:
        return "-"
    return f"[{low:.{precision}f}, {high:.{precision}f}]"


def parse_run_name(run_name: str) -> tuple[str, str] | None:
    """Split a run name into (feature_set, model_type)."""
    for model in sorted(TURN_MODEL_ORDER, key=len, reverse=True):
        if run_name.endswith(f"_{model}"):
            feature_set = run_name[: -(len(model) + 1)]
            return feature_set, model

    multimodal_match = re.match(
        r"^(?P<combo>.+)_turns_multimodal_(?P<backbone>simple|dyadic_shared|attention)_(?P<fusion>gated|concat)$",
        run_name,
    )
    if multimodal_match:
        combo = multimodal_match.group("combo")
        model = f"turns_multimodal_{multimodal_match.group('backbone')}_{multimodal_match.group('fusion')}"
        return combo, model

    multimodal_window_match = re.match(
        r"^(?P<combo>.+)_windows_multimodal_(?P<backbone>simple|dyadic_shared|attention)_(?P<fusion>gated|concat)$",
        run_name,
    )
    if multimodal_window_match:
        combo = multimodal_window_match.group("combo")
        model = f"windows_multimodal_{multimodal_window_match.group('backbone')}_{multimodal_window_match.group('fusion')}"
        return combo, model

    window_match = re.match(r"^(?P<feature_set>.+)_windows_(?P<backbone>simple|dyadic_shared|attention)$", run_name)
    if window_match:
        feature_set = window_match.group("feature_set")
        model = f"windows_{window_match.group('backbone')}"
        return feature_set, model

    return None


def scan_experiments(experiments_dir: Path) -> dict[tuple[str, str], dict]:
    """Scan experiment directories and collect metrics."""
    results: dict[tuple[str, str], dict] = {}
    if not experiments_dir.is_dir():
        return results

    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best_model = run_dir / "model_best.pt"
        if not best_model.exists():
            continue

        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        feature_set, model_type = parsed

        overall = read_metric_csv(run_dir / "metrics_overall.csv")
        role_metrics = read_role_metrics(run_dir / "metrics_by_role.csv")

        results[(feature_set, model_type)] = {
            "run_name": run_dir.name,
            "run_dir": run_dir,
            "ccc": float(overall.get("ccc", "nan")),
            "mae": float(overall.get("mae", "nan")),
            "rmse": float(overall.get("rmse", "nan")),
            "pearson": float(overall.get("pearson", "nan")),
            "novice_ccc": role_metrics.get("novice", float("nan")),
            "expert_ccc": role_metrics.get("expert", float("nan")),
        }

    return results


def fmt(val: float, best: float | None = None, precision: int = 4) -> str:
    """Format a metric value, bolding the best in its column."""
    if val != val:  # NaN
        return "-"
    s = f"{val:.{precision}f}"
    if best is not None and abs(val - best) < 1e-8:
        return f"**{s}**"
    return s


def is_finite(val: float) -> bool:
    """Return True when a float contains a usable numeric result."""

    return not math.isnan(val)


def feature_family(feature_set: str) -> str:
    """Return the high-level modality family for a registered feature set."""

    try:
        return FEATURE_FAMILIES[feature_set]
    except KeyError as exc:
        raise KeyError(f"No family mapping registered for feature set {feature_set!r}.") from exc


def resolve_turn_backbone(results: dict[tuple[str, str], dict], allow_partial_grid: bool = False) -> dict[str, object]:
    """Resolve the winner across unimodal turn results and pick representative streams."""

    win_counts = {model: 0 for model in TURN_MODEL_ORDER}
    feature_winners: list[dict[str, object]] = []
    incomplete_features: list[dict[str, object]] = []

    for feature_set in FEATURE_ORDER:
        entries = []
        missing = []
        for model_type in TURN_MODEL_ORDER:
            record = results.get((feature_set, model_type))
            if record is None or not is_finite(record["ccc"]):
                missing.append(model_type)
                continue
            entries.append((model_type, record["ccc"]))

        if not entries:
            continue
        if missing:
            incomplete_features.append({"feature_set": feature_set, "missing_models": missing})
            if not allow_partial_grid:
                missing_text = ", ".join(missing)
                raise RuntimeError(
                    f"Feature set {feature_set!r} is missing unimodal turn results for: {missing_text}."
                )

        best_ccc = max(ccc for _, ccc in entries)
        winners = [model_type for model_type, ccc in entries if abs(ccc - best_ccc) < 1e-8]
        if len(winners) != 1:
            winner_text = ", ".join(winners)
            raise RuntimeError(f"Feature set {feature_set!r} has a tie between: {winner_text}.")

        winner = winners[0]
        win_counts[winner] += 1
        feature_winners.append(
            {
                "feature_set": feature_set,
                "feature_label": FEATURE_LABELS.get(feature_set, feature_set),
                "winner": winner,
                "winner_label": MODEL_LABELS.get(winner, winner),
                "winner_ccc": best_ccc,
            }
        )

    if not feature_winners:
        raise RuntimeError("No completed unimodal turn results were found for backbone resolution.")

    max_wins = max(win_counts.values())
    tied_backbones = [model for model, count in win_counts.items() if count == max_wins and count > 0]
    if len(tied_backbones) != 1:
        # Break tie by mean CCC across all feature sets.
        def _mean_ccc(model: str) -> float:
            cccs = [
                results[(fs, model)]["ccc"]
                for fs in FEATURE_ORDER
                if (fs, model) in results and is_finite(results[(fs, model)]["ccc"])
            ]
            return sum(cccs) / len(cccs) if cccs else float("-inf")

        tied_backbones.sort(key=_mean_ccc, reverse=True)
        tie_text = ", ".join(f"{m}={win_counts[m]} (mean={_mean_ccc(m):.4f})" for m in tied_backbones)
        print(f"NOTE: Backbone vote tied: {tie_text}. Breaking by mean CCC → {tied_backbones[0]}.")

    backbone_model = tied_backbones[0]
    representatives: dict[str, dict[str, object]] = {}
    for family in FAMILY_ORDER:
        family_rows = []
        for feature_set in FEATURE_ORDER:
            if feature_family(feature_set) != family:
                continue
            record = results.get((feature_set, backbone_model))
            if record is None or not is_finite(record["ccc"]):
                continue
            family_rows.append((feature_set, record["ccc"]))
        if not family_rows:
            continue

        best_feature, best_ccc = max(family_rows, key=lambda row: row[1])
        representatives[family] = {
            "feature_set": best_feature,
            "feature_label": FEATURE_LABELS.get(best_feature, best_feature),
            "ccc": best_ccc,
        }

    missing_families = [family for family in FAMILY_ORDER if family not in representatives]
    if missing_families:
        missing_text = ", ".join(missing_families)
        raise RuntimeError(
            f"Backbone {backbone_model!r} does not have representative results for: {missing_text}."
        )

    return {
        "backbone_model_type": backbone_model,
        "backbone_model_label": MODEL_LABELS.get(backbone_model, backbone_model),
        "backbone_trainer_model": MODEL_TO_TRAINER[backbone_model],
        "backbone_win_count": win_counts[backbone_model],
        "backbone_total_features": len(feature_winners),
        "win_counts": win_counts,
        "feature_winners": feature_winners,
        "representatives": representatives,
        "incomplete_features": incomplete_features,
    }


def print_turn_backbone_resolution(selection: dict[str, object], output_format: str) -> None:
    """Render the resolved unimodal winner in a shell-friendly or human-friendly format."""

    if output_format == "json":
        print(json.dumps(selection, indent=2, sort_keys=True))
        return

    if output_format == "env":
        print(f"TURN_BACKBONE_MODEL_TYPE={selection['backbone_model_type']}")
        print(f"TURN_BACKBONE_TRAINER_MODEL={selection['backbone_trainer_model']}")
        print(f"TURN_BACKBONE_WIN_COUNT={selection['backbone_win_count']}")
        print(f"TURN_BACKBONE_TOTAL_FEATURES={selection['backbone_total_features']}")
        for family in FAMILY_ORDER:
            rep = selection["representatives"][family]
            key = family.upper()
            print(f"BEST_{key}_FEATURE_SET={rep['feature_set']}")
            print(f"BEST_{key}_CCC={rep['ccc']:.6f}")
        return

    print("\n## Turn Backbone Resolution\n")
    print(
        f"Winner: {selection['backbone_model_label']} "
        f"({selection['backbone_model_type']}, trainer=`{selection['backbone_trainer_model']}`)"
    )
    print(
        f"Wins: {selection['backbone_win_count']} / {selection['backbone_total_features']} feature sets\n"
    )

    print("### Win Counts\n")
    print("| Model | Wins |")
    print("|---|---|")
    for model_type in TURN_MODEL_ORDER:
        print(f"| {MODEL_LABELS.get(model_type, model_type)} | {selection['win_counts'][model_type]} |")
    print()

    print("### Representative Feature Sets\n")
    print("| Family | Feature Set | Val CCC |")
    print("|---|---|---|")
    for family in FAMILY_ORDER:
        rep = selection["representatives"][family]
        print(
            f"| {FAMILY_LABELS.get(family, family)} "
            f"| {rep['feature_label']} "
            f"| {rep['ccc']:.4f} |"
        )
    print()

    if selection["incomplete_features"]:
        print("### Incomplete Feature Sets\n")
        print("| Feature Set | Missing Models |")
        print("|---|---|")
        for row in selection["incomplete_features"]:
            missing_text = ", ".join(row["missing_models"])
            print(f"| {FEATURE_LABELS.get(row['feature_set'], row['feature_set'])} | {missing_text} |")
        print()


def print_table_by_model(results: dict[tuple[str, str], dict]) -> None:
    """Print one table per model type: features as rows, metrics as columns."""
    print("\n## Results by Model Type (Validation CCC)\n")

    for model in TURN_MODEL_ORDER:
        model_results = {fs: r for (fs, mt), r in results.items() if mt == model}
        if not model_results:
            continue

        print(f"### {MODEL_LABELS.get(model, model)}\n")
        print("| Feature Set | Val CCC | Novice | Expert | MAE | Organizer MLP (test) |")
        print("|---|---|---|---|---|---|")

        # Find best CCC in this model type for bolding.
        cccs = [r["ccc"] for r in model_results.values() if r["ccc"] == r["ccc"]]
        best_ccc = max(cccs) if cccs else None

        for fs in FEATURE_ORDER:
            if fs not in model_results:
                continue
            r = model_results[fs]
            org = ORGANIZER_BASELINE.get(fs, float("nan"))
            print(
                f"| {FEATURE_LABELS.get(fs, fs)} "
                f"| {fmt(r['ccc'], best_ccc)} "
                f"| {fmt(r['novice_ccc'])} "
                f"| {fmt(r['expert_ccc'])} "
                f"| {fmt(r['mae'])} "
                f"| {fmt(org)} |"
            )
        print()


def print_table_by_feature(results: dict[tuple[str, str], dict]) -> None:
    """Print one summary table: features as rows, model types as columns."""
    print("\n## Summary: Val CCC by Feature Set x Model Type\n")

    # Only include models that have at least one result.
    active_models = [m for m in TURN_MODEL_ORDER if any(mt == m for (_, mt) in results)]
    if not active_models:
        print("No completed experiments found.\n")
        return

    header = "| Feature Set | " + " | ".join(MODEL_LABELS.get(m, m) for m in active_models) + " | Organizer MLP (test) |"
    sep = "|---" * (len(active_models) + 2) + "|"
    print(header)
    print(sep)

    for fs in FEATURE_ORDER:
        row_vals = []
        for m in active_models:
            r = results.get((fs, m))
            row_vals.append(r["ccc"] if r else float("nan"))

        valid = [v for v in row_vals if v == v]
        best = max(valid) if valid else None
        org = ORGANIZER_BASELINE.get(fs, float("nan"))

        cells = " | ".join(fmt(v, best) for v in row_vals)
        print(f"| {FEATURE_LABELS.get(fs, fs)} | {cells} | {fmt(org)} |")

    print()


def print_multimodal_table(results: dict[tuple[str, str], dict]) -> None:
    """Print winner-only multimodal fusion results when present."""

    active_models = sorted(
        {model_type for (_, model_type) in results if model_type.startswith("turns_multimodal_")},
        key=multimodal_model_sort_key,
    )
    if not active_models:
        return

    print("\n## Winner-Only Multimodal Fusion (Validation CCC)\n")
    header = "| Combination | " + " | ".join(model_label(model) for model in active_models) + " |"
    sep = "|---" * (len(active_models) + 1) + "|"
    print(header)
    print(sep)

    combos = sorted({feature_set for (feature_set, model_type) in results if model_type in active_models}, key=lambda item: (item.count("__"), item))
    for combo in combos:
        values = []
        for model_type in active_models:
            record = results.get((combo, model_type))
            values.append(record["ccc"] if record else float("nan"))
        valid = [val for val in values if is_finite(val)]
        best = max(valid) if valid else None
        print(f"| {combo_label(combo)} | {' | '.join(fmt(val, best) for val in values)} |")
    print()


def print_window_comparison_table(results: dict[tuple[str, str], dict]) -> None:
    """Print turn-vs-legacy-window comparisons for any trained winner backbone."""

    window_models = sorted(
        {model_type for (_, model_type) in results if model_type.startswith("windows_")},
        key=lambda item: TRAINER_ORDER.index(item.removeprefix("windows_")) if item.removeprefix("windows_") in TRAINER_ORDER else len(TRAINER_ORDER),
    )
    if not window_models:
        return

    print("\n## Turn vs Legacy Windows (Unimodal Validation CCC)\n")
    for window_model in window_models:
        backbone = window_model.removeprefix("windows_")
        turn_model = TRAINER_TO_TURN_MODEL.get(backbone)
        if turn_model is None:
            continue

        print(f"### {TRAINER_LABELS.get(backbone, backbone)}\n")
        print("| Feature Set | Speech Turns | Legacy Windows | Turn - Window |")
        print("|---|---|---|---|")
        for feature_set in FEATURE_ORDER:
            turn_record = results.get((feature_set, turn_model))
            window_record = results.get((feature_set, window_model))
            if turn_record is None and window_record is None:
                continue

            turn_ccc = turn_record["ccc"] if turn_record else float("nan")
            window_ccc = window_record["ccc"] if window_record else float("nan")
            best = None
            if is_finite(turn_ccc) or is_finite(window_ccc):
                best = max(val for val in [turn_ccc, window_ccc] if is_finite(val))
            delta = turn_ccc - window_ccc if is_finite(turn_ccc) and is_finite(window_ccc) else float("nan")
            print(
                f"| {FEATURE_LABELS.get(feature_set, feature_set)} "
                f"| {fmt(turn_ccc, best)} "
                f"| {fmt(window_ccc, best)} "
                f"| {fmt_delta(delta)} |"
            )
        print()


def print_multimodal_window_comparison_table(results: dict[tuple[str, str], dict]) -> None:
    """Print multimodal speech-turn vs legacy-window comparisons when both exist."""

    turn_models = sorted(
        {model_type for (_, model_type) in results if model_type.startswith("turns_multimodal_")},
        key=multimodal_model_sort_key,
    )
    window_models = {
        model_type for (_, model_type) in results if model_type.startswith("windows_multimodal_")
    }
    if not turn_models or not window_models:
        return

    printed_any = False
    print("\n## Turn vs Legacy Windows (Multimodal Validation CCC)\n")
    for turn_model in turn_models:
        suffix = turn_model.removeprefix("turns_multimodal_")
        window_model = f"windows_multimodal_{suffix}"
        if window_model not in window_models:
            continue

        printed_any = True
        print(f"### {model_label(turn_model)}\n")
        print("| Combination | Speech Turns | Legacy Windows | Turn - Window |")
        print("|---|---|---|---|")

        combos = sorted(
            {
                feature_set
                for (feature_set, model_type) in results
                if model_type in {turn_model, window_model}
            },
            key=lambda item: (item.count("__"), item),
        )
        for combo in combos:
            turn_record = results.get((combo, turn_model))
            window_record = results.get((combo, window_model))
            if turn_record is None and window_record is None:
                continue

            turn_ccc = turn_record["ccc"] if turn_record else float("nan")
            window_ccc = window_record["ccc"] if window_record else float("nan")
            best = None
            if is_finite(turn_ccc) or is_finite(window_ccc):
                best = max(val for val in [turn_ccc, window_ccc] if is_finite(val))
            delta = turn_ccc - window_ccc if is_finite(turn_ccc) and is_finite(window_ccc) else float("nan")
            print(
                f"| {combo_label(combo)} "
                f"| {fmt(turn_ccc, best)} "
                f"| {fmt(window_ccc, best)} "
                f"| {fmt_delta(delta)} |"
            )
        print()

    if not printed_any:
        print("No completed multimodal legacy-window comparisons found.\n")


def bootstrap_mean_ci(
    diffs: list[float],
    rng: random.Random,
    n_resamples: int,
) -> tuple[float, float]:
    """Bootstrap percentile interval for the mean paired delta."""

    if not diffs:
        return float("nan"), float("nan")
    if len(diffs) == 1:
        return diffs[0], diffs[0]

    n = len(diffs)
    means: list[float] = []
    for _ in range(max(1, n_resamples)):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = max(0, int(0.025 * (len(means) - 1)))
    high_idx = min(len(means) - 1, int(0.975 * (len(means) - 1)))
    return means[low_idx], means[high_idx]


def sign_flip_p_value(
    diffs: list[float],
    rng: random.Random,
    n_samples: int,
    exact_limit: int,
) -> tuple[float, str]:
    """One-sided paired sign-flip test for mean(delta) > 0."""

    nonzero = [diff for diff in diffs if abs(diff) > 1e-12]
    if not nonzero:
        return 1.0, "degenerate"

    observed = sum(nonzero) / len(nonzero)
    magnitudes = [abs(diff) for diff in nonzero]
    n = len(magnitudes)

    if n <= exact_limit:
        ge_count = 0
        total = 1 << n
        for mask in range(total):
            signed_sum = 0.0
            for idx, magnitude in enumerate(magnitudes):
                signed_sum += magnitude if ((mask >> idx) & 1) else -magnitude
            if (signed_sum / n) >= observed - 1e-12:
                ge_count += 1
        return ge_count / total, f"exact_sign_flip(n={n})"

    ge_count = 0
    draws = max(1, n_samples)
    for _ in range(draws):
        signed_sum = 0.0
        for magnitude in magnitudes:
            signed_sum += magnitude if rng.random() >= 0.5 else -magnitude
        if (signed_sum / n) >= observed - 1e-12:
            ge_count += 1
    return (ge_count + 1) / (draws + 1), f"monte_carlo_sign_flip(n={n}, samples={draws})"


def paired_multimodal_significance_rows(
    results: dict[tuple[str, str], dict],
    *,
    alpha: float,
    permutation_samples: int,
    bootstrap_samples: int,
    exact_limit: int,
    seed: int,
) -> list[dict[str, object]]:
    """Compute paired session-level significance rows for multimodal turn vs window runs."""

    turn_models = sorted(
        {model_type for (_, model_type) in results if model_type.startswith("turns_multimodal_")},
        key=multimodal_model_sort_key,
    )
    rows: list[dict[str, object]] = []

    for turn_model in turn_models:
        suffix = turn_model.removeprefix("turns_multimodal_")
        window_model = f"windows_multimodal_{suffix}"

        combos = sorted(
            {
                feature_set
                for (feature_set, model_type) in results
                if model_type in {turn_model, window_model}
            },
            key=lambda item: (item.count("__"), item),
        )
        for combo in combos:
            turn_record = results.get((combo, turn_model))
            window_record = results.get((combo, window_model))
            if turn_record is None or window_record is None:
                continue

            turn_sessions = read_session_metrics(turn_record["run_dir"] / "metrics_by_session.csv")
            window_sessions = read_session_metrics(window_record["run_dir"] / "metrics_by_session.csv")
            shared_keys = sorted(set(turn_sessions) & set(window_sessions))
            if not shared_keys:
                continue

            diffs: list[float] = []
            positive = 0
            negative = 0
            ties = 0
            dropped_nonfinite = 0
            for key in shared_keys:
                turn_ccc = turn_sessions[key]["ccc"]
                window_ccc = window_sessions[key]["ccc"]
                if not (is_finite(turn_ccc) and is_finite(window_ccc)):
                    dropped_nonfinite += 1
                    continue
                diff = turn_ccc - window_ccc
                diffs.append(diff)
                if diff > 1e-12:
                    positive += 1
                elif diff < -1e-12:
                    negative += 1
                else:
                    ties += 1

            if not diffs:
                continue

            bootstrap_rng = random.Random(f"{seed}:{turn_model}:{combo}:bootstrap")
            permutation_rng = random.Random(f"{seed}:{turn_model}:{combo}:permutation")
            mean_delta = sum(diffs) / len(diffs)
            sorted_diffs = sorted(diffs)
            mid = len(sorted_diffs) // 2
            if len(sorted_diffs) % 2 == 1:
                median_delta = sorted_diffs[mid]
            else:
                median_delta = 0.5 * (sorted_diffs[mid - 1] + sorted_diffs[mid])
            ci_low, ci_high = bootstrap_mean_ci(diffs, bootstrap_rng, n_resamples=bootstrap_samples)
            p_value, method = sign_flip_p_value(
                diffs,
                permutation_rng,
                n_samples=permutation_samples,
                exact_limit=exact_limit,
            )

            rows.append(
                {
                    "model_label": model_label(turn_model),
                    "combination": combo,
                    "combination_label": combo_label(combo),
                    "n_sessions": len(diffs),
                    "shared_sessions": len(shared_keys),
                    "dropped_nonfinite": dropped_nonfinite,
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "positive": positive,
                    "negative": negative,
                    "ties": ties,
                    "p_value": p_value,
                    "significant": p_value < alpha and mean_delta > 0.0,
                    "method": method,
                }
            )

    return rows


def print_multimodal_significance_table(
    results: dict[tuple[str, str], dict],
    *,
    alpha: float,
    permutation_samples: int,
    bootstrap_samples: int,
    exact_limit: int,
    seed: int,
) -> None:
    """Print a paired significance table for multimodal speech-turn vs window runs."""

    turn_models = {model_type for (_, model_type) in results if model_type.startswith("turns_multimodal_")}
    window_models = {model_type for (_, model_type) in results if model_type.startswith("windows_multimodal_")}
    has_candidate_pairs = any(
        f"windows_multimodal_{turn_model.removeprefix('turns_multimodal_')}" in window_models
        for turn_model in turn_models
    )

    rows = paired_multimodal_significance_rows(
        results,
        alpha=alpha,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
        exact_limit=exact_limit,
        seed=seed,
    )
    if not rows:
        if has_candidate_pairs:
            print("\n## Significance: Speech Turns > Legacy Windows (Multimodal)\n")
            print("No paired session metrics were available for multimodal turn-vs-window significance testing.\n")
        return

    print("\n## Significance: Speech Turns > Legacy Windows (Multimodal)\n")
    print("Paired test over session-level CCC differences. Positive deltas favor speech turns.")
    print()
    print("| Model | Combination | Sessions | Mean Delta CCC | 95% CI | Wins | Losses | Ties | p(one-sided) | Significant |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        significant_text = "yes" if row["significant"] else "no"
        print(
            f"| {row['model_label']} "
            f"| {row['combination_label']} "
            f"| {row['n_sessions']} "
            f"| {fmt_delta(row['mean_delta'])} "
            f"| {fmt_interval(row['ci_low'], row['ci_high'])} "
            f"| {row['positive']} "
            f"| {row['negative']} "
            f"| {row['ties']} "
            f"| {row['p_value']:.4g} "
            f"| {significant_text} |"
        )
    print()
    print(f"Alpha = {alpha:.3f}. Test = paired sign-flip permutation (exact when n <= {exact_limit}).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and format ACM experiment results.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments",
    )
    parser.add_argument(
        "--resolve-turn-backbone",
        action="store_true",
        help="Resolve the majority-winning unimodal turn backbone and representative audio/text/visual streams.",
    )
    parser.add_argument(
        "--selection-format",
        choices=["markdown", "env", "json"],
        default="markdown",
        help="Output format used together with --resolve-turn-backbone.",
    )
    parser.add_argument(
        "--allow-partial-grid",
        action="store_true",
        help="Allow backbone selection when some feature sets are missing one or more unimodal turn runs.",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=DEFAULT_SIGNIFICANCE_ALPHA,
        help="Alpha threshold for the multimodal turn-vs-window significance section.",
    )
    parser.add_argument(
        "--significance-permutations",
        type=int,
        default=DEFAULT_PERMUTATION_SAMPLES,
        help="Number of Monte Carlo sign-flip samples when exact enumeration is too large.",
    )
    parser.add_argument(
        "--significance-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples for the mean delta confidence interval.",
    )
    parser.add_argument(
        "--significance-exact-limit",
        type=int,
        default=DEFAULT_EXACT_LIMIT,
        help="Use exact sign-flip enumeration when paired session count is at most this value.",
    )
    parser.add_argument(
        "--significance-seed",
        type=int,
        default=PERMUTATION_SEED,
        help="Random seed used by Monte Carlo sign-flip and bootstrap resampling.",
    )
    args = parser.parse_args()

    results = scan_experiments(args.experiments_dir)
    if not results:
        print(f"No completed experiments found in {args.experiments_dir}")
        sys.exit(1)

    if args.resolve_turn_backbone:
        selection = resolve_turn_backbone(results, allow_partial_grid=args.allow_partial_grid)
        print_turn_backbone_resolution(selection, output_format=args.selection_format)
        return

    n = len(results)
    features = sorted({fs for fs, _ in results})
    models = sorted({mt for _, mt in results})
    print(f"Found {n} completed experiments across {len(features)} feature sets and {len(models)} model types.\n")

    print_table_by_feature(results)
    print_table_by_model(results)
    print_multimodal_table(results)
    print_window_comparison_table(results)
    print_multimodal_window_comparison_table(results)
    print_multimodal_significance_table(
        results,
        alpha=args.significance_alpha,
        permutation_samples=args.significance_permutations,
        bootstrap_samples=args.significance_bootstrap,
        exact_limit=args.significance_exact_limit,
        seed=args.significance_seed,
    )


if __name__ == "__main__":
    main()
