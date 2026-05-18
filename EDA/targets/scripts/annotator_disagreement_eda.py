from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
import re


DATA_ROOT = Path("X:/")
PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SPLITS = ("train-cc", "train-cr")
COLORS = ("purple", "yellow")
TASKS = ("task_engagement", "social_engagement")
ANNOT_NUM_RE = re.compile(r"\.(\d+)\.annotation\.csv$")


def read_rows(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle]


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, str]] = []
    session_rows: list[dict[str, str]] = []
    pair_counter: Counter = Counter()
    label_counter: Counter = Counter()
    entropy_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for split in SPLITS:
        split_dir = DATA_ROOT / split
        sessions = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for session in sessions:
            for color in COLORS:
                for task in TASKS:
                    main_file = session / f"{color}.{task}.annotation.csv"
                    if not main_file.exists():
                        continue
                    main_rows = read_rows(main_file)
                    blank_indices = [i for i, v in enumerate(main_rows) if v == ""]
                    if not blank_indices:
                        continue

                    annot_files = sorted(session.glob(f"{color}.{task}.*.annotation.csv"))
                    annot_files = [p for p in annot_files if ANNOT_NUM_RE.search(p.name)]
                    annot_rows = [read_rows(p) for p in annot_files]
                    valid_ann_files = len(annot_rows)
                    if valid_ann_files == 0:
                        continue

                    session_blank_with_votes = 0
                    session_vote_counts = Counter()
                    for idx in blank_indices:
                        votes = []
                        for rows in annot_rows:
                            if idx < len(rows):
                                label = rows[idx].strip()
                                if label != "":
                                    votes.append(label)
                        if not votes:
                            continue
                        session_blank_with_votes += 1
                        for v in votes:
                            session_vote_counts[v] += 1
                            label_counter[(split, color, task, v)] += 1

                        vote_key = " | ".join(sorted(votes))
                        frame_rows.append(
                            {
                                "split": split,
                                "session_id": session.name,
                                "color": color,
                                "task": task,
                                "frame_idx": str(idx),
                                "n_annotator_files": str(valid_ann_files),
                                "n_votes_present": str(len(votes)),
                                "vote_signature": vote_key,
                            }
                        )
                        pair_counter[(split, color, task, vote_key)] += 1

                        counts = Counter(votes)
                        total = sum(counts.values())
                        # entropy in bits for per-frame vote distribution
                        entropy = 0.0
                        for c in counts.values():
                            p = c / total
                            if p > 0:
                                from math import log2
                                entropy -= p * log2(p)
                        entropy_by_group[(split, color, task)].append(entropy)

                    total_blanks = len(blank_indices)
                    blank_vote_rate = (session_blank_with_votes / total_blanks) if total_blanks > 0 else 0.0
                    top_labels = session_vote_counts.most_common(3)
                    top_labels_str = "; ".join([f"{k}:{v}" for k, v in top_labels])
                    session_rows.append(
                        {
                            "split": split,
                            "session_id": session.name,
                            "color": color,
                            "task": task,
                            "blank_rows_main": str(total_blanks),
                            "blank_rows_with_votes": str(session_blank_with_votes),
                            "blank_vote_rate": f"{blank_vote_rate:.8f}",
                            "annotator_file_count": str(valid_ann_files),
                            "top_vote_labels": top_labels_str,
                        }
                    )

    with (OUTPUT_DIR / "annotator_disagreement_frame_votes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "color",
                "task",
                "frame_idx",
                "n_annotator_files",
                "n_votes_present",
                "vote_signature",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    with (OUTPUT_DIR / "annotator_disagreement_session_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "color",
                "task",
                "blank_rows_main",
                "blank_rows_with_votes",
                "blank_vote_rate",
                "annotator_file_count",
                "top_vote_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(session_rows)

    with (OUTPUT_DIR / "annotator_disagreement_signature_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "color", "task", "vote_signature", "count"],
        )
        writer.writeheader()
        for (split, color, task, sig), count in sorted(pair_counter.items()):
            writer.writerow(
                {
                    "split": split,
                    "color": color,
                    "task": task,
                    "vote_signature": sig,
                    "count": str(count),
                }
            )

    with (OUTPUT_DIR / "annotator_disagreement_label_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "color", "task", "label", "count"],
        )
        writer.writeheader()
        for (split, color, task, label), count in sorted(label_counter.items()):
            writer.writerow(
                {"split": split, "color": color, "task": task, "label": label, "count": str(count)}
            )

    with (OUTPUT_DIR / "annotator_disagreement_entropy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "color", "task", "n_frames", "mean_entropy_bits"],
        )
        writer.writeheader()
        for key, values in sorted(entropy_by_group.items()):
            split, color, task = key
            n = len(values)
            mean_ent = sum(values) / n if n > 0 else 0.0
            writer.writerow(
                {
                    "split": split,
                    "color": color,
                    "task": task,
                    "n_frames": str(n),
                    "mean_entropy_bits": f"{mean_ent:.8f}",
                }
            )

    print(f"Wrote annotator disagreement outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
ANNOT_NUM_RE = re.compile(r"\.(\d+)\.annotation\.csv$")
