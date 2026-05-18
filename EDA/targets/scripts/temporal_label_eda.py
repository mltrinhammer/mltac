from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


DATA_ROOT = Path("X:/")
PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SCENARIO = "drop_blank_and_nan"
SPLITS = ("train-cc", "train-cr")
TASKS = ("task_engagement", "social_engagement")
COLOR = "purple"
FRAME_RATE = 30.0
PERSISTENCE_THRESHOLDS = [0.5, 1, 2, 3, 5, 10, 20, 30, 60]


def is_nan(value: str) -> bool:
    return value.strip().lower() == "nan"


def keep_row(value: str) -> bool:
    v = value.strip()
    if SCENARIO == "all":
        return True
    if SCENARIO == "drop_blank":
        return v != ""
    if SCENARIO == "drop_nan":
        return not is_nan(v)
    if SCENARIO == "drop_blank_and_nan":
        return (v != "") and (not is_nan(v))
    raise ValueError(f"Unknown scenario: {SCENARIO}")


def entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def read_rows(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle]


def build_runs(rows: list[str]) -> tuple[list[dict], list[tuple[str, str, int, str]], list[str]]:
    runs: list[dict] = []
    gap_contexts: list[tuple[str, str, int, str]] = []
    filtered_sequence: list[str] = []

    i = 0
    n = len(rows)
    while i < n:
        if keep_row(rows[i]):
            filtered_sequence.append(rows[i])
        i += 1

    # run extraction on filtered sequence
    if filtered_sequence:
        start = 0
        current = filtered_sequence[0]
        run_id = 0
        for idx in range(1, len(filtered_sequence) + 1):
            if idx == len(filtered_sequence) or filtered_sequence[idx] != current:
                end = idx - 1
                length_frames = end - start + 1
                runs.append(
                    {
                        "run_id": run_id,
                        "label": current,
                        "start_idx": start,
                        "end_idx": end,
                        "length_frames": length_frames,
                        "length_seconds": length_frames / FRAME_RATE,
                        "prev_label": runs[-1]["label"] if runs else "<START>",
                        "next_label": "<END>",
                    }
                )
                if runs and len(runs) > 1:
                    runs[-2]["next_label"] = current
                run_id += 1
                if idx < len(filtered_sequence):
                    current = filtered_sequence[idx]
                    start = idx

    # gap contexts on original rows
    idx = 0
    while idx < n:
        if keep_row(rows[idx]):
            idx += 1
            continue
        gap_start = idx
        gap_types = []
        while idx < n and (not keep_row(rows[idx])):
            value = rows[idx].strip()
            if value == "":
                gap_types.append("blank")
            elif is_nan(value):
                gap_types.append("nan")
            else:
                gap_types.append("other_removed")
            idx += 1
        gap_end = idx - 1
        prev_label = "<START>"
        next_label = "<END>"
        p = gap_start - 1
        while p >= 0:
            if keep_row(rows[p]):
                prev_label = rows[p]
                break
            p -= 1
        q = gap_end + 1
        while q < n:
            if keep_row(rows[q]):
                next_label = rows[q]
                break
            q += 1
        gap_len = gap_end - gap_start + 1
        gap_type = "mixed" if len(set(gap_types)) > 1 else (gap_types[0] if gap_types else "unknown")
        gap_contexts.append((prev_label, next_label, gap_len, gap_type))

    return runs, gap_contexts, filtered_sequence


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, str]] = []
    by_label_lengths: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_label_counts: Counter = Counter()
    by_session_summary: list[dict[str, str]] = []
    transition_counts: Counter = Counter()
    gap_rows: list[dict[str, str]] = []
    session_heterogeneity_rows: list[dict[str, str]] = []

    for split in SPLITS:
        split_dir = DATA_ROOT / split
        sessions = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for session in sessions:
            for task in TASKS:
                label_file = session / f"{COLOR}.{task}.annotation.csv"
                if not label_file.exists():
                    continue
                rows = read_rows(label_file)
                runs, gap_contexts, filtered_sequence = build_runs(rows)

                session_label_counter = Counter()
                session_transition_count = 0
                total_seconds = len(filtered_sequence) / FRAME_RATE
                for r in runs:
                    key = (split, task, r["label"])
                    by_label_lengths[key].append(r["length_seconds"])
                    by_label_counts[(split, task, r["label"])] += 1
                    session_label_counter[r["label"]] += r["length_frames"]
                    run_rows.append(
                        {
                            "split": split,
                            "session_id": session.name,
                            "task": task,
                            "label": r["label"],
                            "run_id": str(r["run_id"]),
                            "start_idx": str(r["start_idx"]),
                            "end_idx": str(r["end_idx"]),
                            "length_frames": str(r["length_frames"]),
                            "length_seconds": f"{r['length_seconds']:.8f}",
                            "prev_label": r["prev_label"],
                            "next_label": r["next_label"],
                        }
                    )
                    if r["next_label"] not in ("<END>", r["label"]):
                        transition_counts[(split, task, r["label"], r["next_label"])] += 1
                        session_transition_count += 1

                for prev_label, next_label, gap_len, gap_type in gap_contexts:
                    gap_rows.append(
                        {
                            "split": split,
                            "session_id": session.name,
                            "task": task,
                            "left_label": prev_label,
                            "right_label": next_label,
                            "gap_len_frames": str(gap_len),
                            "gap_type": gap_type,
                        }
                    )

                mean_run_sec = (sum(r["length_seconds"] for r in runs) / len(runs)) if runs else 0.0
                median_run_sec = quantile([r["length_seconds"] for r in runs], 0.5) if runs else 0.0
                transition_rate_per_min = (session_transition_count / total_seconds) * 60.0 if total_seconds > 0 else 0.0
                label_entropy = entropy_from_counts(session_label_counter)
                session_heterogeneity_rows.append(
                    {
                        "split": split,
                        "session_id": session.name,
                        "task": task,
                        "transition_rate_per_min": f"{transition_rate_per_min:.8f}",
                        "mean_run_sec": f"{mean_run_sec:.8f}",
                        "median_run_sec": f"{median_run_sec:.8f}",
                        "entropy_label_mix": f"{label_entropy:.8f}",
                    }
                )

    # 1) Runs by session
    with (OUTPUT_DIR / "label_runs_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "task",
                "label",
                "run_id",
                "start_idx",
                "end_idx",
                "length_frames",
                "length_seconds",
                "prev_label",
                "next_label",
            ],
        )
        writer.writeheader()
        writer.writerows(run_rows)

    # 2) Run summary by label
    label_summary_rows: list[dict[str, str]] = []
    for (split, task, label), lengths in sorted(by_label_lengths.items()):
        n_runs = len(lengths)
        total_seconds = sum(lengths)
        total_frames = int(round(total_seconds * FRAME_RATE))
        label_summary_rows.append(
            {
                "split": split,
                "task": task,
                "label": label,
                "n_runs": str(n_runs),
                "total_frames": str(total_frames),
                "total_seconds": f"{total_seconds:.8f}",
                "mean_sec": f"{(total_seconds / n_runs):.8f}" if n_runs > 0 else "0.00000000",
                "median_sec": f"{quantile(lengths, 0.5):.8f}" if n_runs > 0 else "0.00000000",
                "p25_sec": f"{quantile(lengths, 0.25):.8f}" if n_runs > 0 else "0.00000000",
                "p75_sec": f"{quantile(lengths, 0.75):.8f}" if n_runs > 0 else "0.00000000",
                "max_sec": f"{max(lengths):.8f}" if n_runs > 0 else "0.00000000",
            }
        )
    with (OUTPUT_DIR / "label_run_summary_by_label.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "task",
                "label",
                "n_runs",
                "total_frames",
                "total_seconds",
                "mean_sec",
                "median_sec",
                "p25_sec",
                "p75_sec",
                "max_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(label_summary_rows)

    # 3) Run summary by session
    with (OUTPUT_DIR / "label_run_summary_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "task",
                "n_runs",
                "total_seconds",
                "mean_run_sec",
                "median_run_sec",
                "unique_labels",
            ],
        )
        writer.writeheader()
        for row in session_heterogeneity_rows:
            pass
        # derive from run rows
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        labels_seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for r in run_rows:
            key = (r["split"], r["session_id"], r["task"])
            grouped[key].append(float(r["length_seconds"]))
            labels_seen[key].add(r["label"])
        out = []
        for (split, session_id, task), lengths in sorted(grouped.items()):
            out.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "task": task,
                    "n_runs": str(len(lengths)),
                    "total_seconds": f"{sum(lengths):.8f}",
                    "mean_run_sec": f"{(sum(lengths)/len(lengths)):.8f}" if lengths else "0.00000000",
                    "median_run_sec": f"{quantile(lengths, 0.5):.8f}" if lengths else "0.00000000",
                    "unique_labels": str(len(labels_seen[(split, session_id, task)])),
                }
            )
        writer.writerows(out)

    # 4) transition counts/probs
    with (OUTPUT_DIR / "label_transition_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "task", "from_label", "to_label", "count"])
        writer.writeheader()
        for (split, task, from_label, to_label), count in sorted(transition_counts.items()):
            writer.writerow(
                {
                    "split": split,
                    "task": task,
                    "from_label": from_label,
                    "to_label": to_label,
                    "count": str(count),
                }
            )

    by_from_totals: Counter = Counter()
    for (split, task, from_label, _), count in transition_counts.items():
        by_from_totals[(split, task, from_label)] += count

    with (OUTPUT_DIR / "label_transition_probs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "task", "from_label", "to_label", "prob"])
        writer.writeheader()
        for (split, task, from_label, to_label), count in sorted(transition_counts.items()):
            denom = by_from_totals[(split, task, from_label)]
            writer.writerow(
                {
                    "split": split,
                    "task": task,
                    "from_label": from_label,
                    "to_label": to_label,
                    "prob": f"{(count / denom):.8f}" if denom > 0 else "0.00000000",
                }
            )

    # 5) gap context
    with (OUTPUT_DIR / "label_transition_with_gap_context.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "session_id", "task", "left_label", "right_label", "gap_len_frames", "gap_type"],
        )
        writer.writeheader()
        writer.writerows(gap_rows)

    # 6) persistence curve
    with (OUTPUT_DIR / "label_persistence_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "task", "label", "t_seconds", "survival_prob"])
        writer.writeheader()
        for (split, task, label), lengths in sorted(by_label_lengths.items()):
            n = len(lengths)
            for t in PERSISTENCE_THRESHOLDS:
                survive = sum(1 for x in lengths if x >= t)
                writer.writerow(
                    {
                        "split": split,
                        "task": task,
                        "label": label,
                        "t_seconds": f"{t:.2f}",
                        "survival_prob": f"{(survive / n):.8f}" if n > 0 else "0.00000000",
                    }
                )

    # 7) temporal heterogeneity
    with (OUTPUT_DIR / "label_temporal_heterogeneity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "session_id", "task", "transition_rate_per_min", "mean_run_sec", "median_run_sec", "entropy_label_mix"],
        )
        writer.writeheader()
        writer.writerows(session_heterogeneity_rows)

    print(f"Wrote temporal label EDA outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()

