from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("X:/")
PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class FileStats:
    total_rows: int = 0
    nonblank_rows: int = 0
    blank_rows: int = 0


SCENARIOS = ("all", "drop_blank", "drop_nan", "drop_blank_and_nan")


def read_label_rows(file_path: Path) -> list[str]:
    rows: list[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            rows.append(raw.strip())
    return rows


def parse_main_label_file(file_path: Path) -> tuple[FileStats, Counter]:
    stats = FileStats()
    counts: Counter = Counter()
    for row in read_label_rows(file_path):
        stats.total_rows += 1
        if row == "":
            stats.blank_rows += 1
            continue
        stats.nonblank_rows += 1
        counts[row] += 1
    return stats, counts


def parse_numeric_annotation_file(file_path: Path) -> tuple[int, Counter]:
    total = 0
    counts: Counter = Counter()
    for row in read_label_rows(file_path):
        if row == "":
            continue
        total += 1
        counts[row] += 1
    return total, counts


def collect_sessions(split_dir: Path) -> list[Path]:
    return sorted([path for path in split_dir.iterdir() if path.is_dir()])


def is_nan_token(value: str) -> bool:
    return value.strip().lower() == "nan"


def keep_row_for_scenario(value: str, scenario: str) -> bool:
    is_blank = value.strip() == ""
    is_nan = is_nan_token(value)
    if scenario == "all":
        return True
    if scenario == "drop_blank":
        return not is_blank
    if scenario == "drop_nan":
        return not is_nan
    if scenario == "drop_blank_and_nan":
        return (not is_blank) and (not is_nan)
    raise ValueError(f"Unknown scenario: {scenario}")


def normalized_class_weights(counts: Counter) -> dict[str, float]:
    labels = [label for label, count in counts.items() if count > 0]
    k = len(labels)
    n = sum(counts[label] for label in labels)
    if k == 0 or n == 0:
        return {}
    return {label: n / (k * counts[label]) for label in labels}


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_to_sessions = {
        "train-cc": collect_sessions(DATA_ROOT / "train-cc"),
        "train-cr": collect_sessions(DATA_ROOT / "train-cr"),
    }

    summary_rows: list[dict[str, str]] = []
    class_rows: list[dict[str, str]] = []
    numeric_rows: list[dict[str, str]] = []
    aggregate_main_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    scenario_class_counts: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    scenario_totals: dict[tuple[str, str, str, str], int] = defaultdict(int)
    scenario_label_rows: list[dict[str, str]] = []
    purple_compare_rows: list[dict[str, str]] = []
    disagreement_rows: list[dict[str, str]] = []
    disagreement_neighbor_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    nan_rows: list[dict[str, str]] = []

    for split, sessions in split_to_sessions.items():
        for session in sessions:
            session_id = session.name
            for color in ("purple", "yellow"):
                for task in ("task_engagement", "social_engagement"):
                    main_file = session / f"{color}.{task}.annotation.csv"
                    if not main_file.exists():
                        continue

                    stats, class_counts = parse_main_label_file(main_file)
                    summary_rows.append(
                        {
                            "split": split,
                            "session_id": session_id,
                            "color": color,
                            "task": task,
                            "file_type": "main",
                            "total_rows": str(stats.total_rows),
                            "nonblank_rows": str(stats.nonblank_rows),
                            "blank_rows": str(stats.blank_rows),
                            "agreement_rate": (
                                f"{(stats.nonblank_rows / stats.total_rows):.6f}"
                                if stats.total_rows > 0
                                else "0.000000"
                            ),
                        }
                    )

                    aggregate_main_counts[(split, color, task)].update(class_counts)
                    for label, count in sorted(class_counts.items()):
                        class_rows.append(
                            {
                                "split": split,
                                "session_id": session_id,
                                "color": color,
                                "task": task,
                                "label": label,
                                "count": str(count),
                            }
                        )

                    rows = read_label_rows(main_file)
                    blank_idx = [i for i, row in enumerate(rows) if row.strip() == ""]
                    nan_idx = [i for i, row in enumerate(rows) if is_nan_token(row)]
                    disagreement_rows.append(
                        {
                            "split": split,
                            "session_id": session_id,
                            "color": color,
                            "task": task,
                            "total_rows": str(len(rows)),
                            "blank_rows": str(len(blank_idx)),
                            "blank_rate": f"{(len(blank_idx) / len(rows)):.8f}" if len(rows) > 0 else "0.00000000",
                        }
                    )
                    nan_rows.append(
                        {
                            "split": split,
                            "session_id": session_id,
                            "color": color,
                            "task": task,
                            "total_rows": str(len(rows)),
                            "nan_rows": str(len(nan_idx)),
                            "nan_rate": f"{(len(nan_idx) / len(rows)):.8f}" if len(rows) > 0 else "0.00000000",
                        }
                    )
                    for idx in blank_idx:
                        prev_label = "<START>"
                        next_label = "<END>"
                        prev_i = idx - 1
                        while prev_i >= 0:
                            if rows[prev_i].strip() != "":
                                prev_label = rows[prev_i].strip()
                                break
                            prev_i -= 1
                        next_i = idx + 1
                        while next_i < len(rows):
                            if rows[next_i].strip() != "":
                                next_label = rows[next_i].strip()
                                break
                            next_i += 1
                        disagreement_neighbor_counts[(split, color, task, prev_label, next_label)] += 1

                    for scenario in SCENARIOS:
                        filtered: list[str] = [row for row in rows if keep_row_for_scenario(row, scenario)]
                        counts = Counter(filtered)
                        key = (scenario, split, color, task)
                        scenario_class_counts[key].update(counts)
                        scenario_totals[key] += len(filtered)
                        for label, count in sorted(counts.items()):
                            scenario_label_rows.append(
                                {
                                    "scenario": scenario,
                                    "split": split,
                                    "session_id": session_id,
                                    "color": color,
                                    "task": task,
                                    "label": label if label != "" else "<blank>",
                                    "count": str(count),
                                }
                            )

                    numeric_candidates = sorted(session.glob(f"{color}.{task}.*.annotation.csv"))
                    for numeric_file in numeric_candidates:
                        suffix = numeric_file.name.replace(f"{color}.{task}.", "")
                        if suffix == "annotation.csv":
                            continue
                        total, numeric_counts = parse_numeric_annotation_file(numeric_file)
                        numeric_rows.append(
                            {
                                "split": split,
                                "session_id": session_id,
                                "color": color,
                                "task": task,
                                "annotator_file": numeric_file.name,
                                "rows_nonblank": str(total),
                                "distinct_labels": str(len(numeric_counts)),
                            }
                        )

    with (OUTPUT_DIR / "label_summary_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "color",
                "task",
                "file_type",
                "total_rows",
                "nonblank_rows",
                "blank_rows",
                "agreement_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUTPUT_DIR / "label_class_counts_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "session_id", "color", "task", "label", "count"],
        )
        writer.writeheader()
        writer.writerows(class_rows)

    with (OUTPUT_DIR / "label_numeric_annotation_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "color",
                "task",
                "annotator_file",
                "rows_nonblank",
                "distinct_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(numeric_rows)

    with (OUTPUT_DIR / "label_aggregate_main_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "color", "task", "label", "count"],
        )
        writer.writeheader()
        for (split, color, task), counter in sorted(aggregate_main_counts.items()):
            for label, count in sorted(counter.items()):
                writer.writerow(
                    {
                        "split": split,
                        "color": color,
                        "task": task,
                        "label": label,
                        "count": str(count),
                    }
                )

    with (OUTPUT_DIR / "label_scenario_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "split", "session_id", "color", "task", "label", "count"],
        )
        writer.writeheader()
        writer.writerows(scenario_label_rows)

    with (OUTPUT_DIR / "label_scenario_aggregate_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "split", "color", "task", "label", "count", "proportion"],
        )
        writer.writeheader()
        for key, counter in sorted(scenario_class_counts.items()):
            scenario, split, color, task = key
            total = scenario_totals[key]
            for label, count in sorted(counter.items()):
                writer.writerow(
                    {
                        "scenario": scenario,
                        "split": split,
                        "color": color,
                        "task": task,
                        "label": label if label != "" else "<blank>",
                        "count": str(count),
                        "proportion": f"{(count / total):.8f}" if total > 0 else "0.00000000",
                    }
                )

    with (OUTPUT_DIR / "label_scenario_class_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "split", "color", "task", "label", "count", "weight"],
        )
        writer.writeheader()
        for key, counter in sorted(scenario_class_counts.items()):
            scenario, split, color, task = key
            weights = normalized_class_weights(counter)
            for label, weight in sorted(weights.items()):
                writer.writerow(
                    {
                        "scenario": scenario,
                        "split": split,
                        "color": color,
                        "task": task,
                        "label": label if label != "" else "<blank>",
                        "count": str(counter[label]),
                        "weight": f"{weight:.8f}",
                    }
                )

    for scenario in SCENARIOS:
        for task in ("task_engagement", "social_engagement"):
            cc_key = (scenario, "train-cc", "purple", task)
            cr_key = (scenario, "train-cr", "purple", task)
            cc_counts = scenario_class_counts.get(cc_key, Counter())
            cr_counts = scenario_class_counts.get(cr_key, Counter())
            cc_total = scenario_totals.get(cc_key, 0)
            cr_total = scenario_totals.get(cr_key, 0)
            labels = sorted(set(cc_counts.keys()).union(set(cr_counts.keys())))
            for label in labels:
                cc_count = cc_counts.get(label, 0)
                cr_count = cr_counts.get(label, 0)
                cc_prop = (cc_count / cc_total) if cc_total > 0 else 0.0
                cr_prop = (cr_count / cr_total) if cr_total > 0 else 0.0
                purple_compare_rows.append(
                    {
                        "scenario": scenario,
                        "task": task,
                        "label": label if label != "" else "<blank>",
                        "cc_count": str(cc_count),
                        "cr_count": str(cr_count),
                        "cc_proportion": f"{cc_prop:.8f}",
                        "cr_proportion": f"{cr_prop:.8f}",
                        "proportion_diff_cc_minus_cr": f"{(cc_prop - cr_prop):.8f}",
                    }
                )

    with (OUTPUT_DIR / "purple_cc_vs_cr_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "task",
                "label",
                "cc_count",
                "cr_count",
                "cc_proportion",
                "cr_proportion",
                "proportion_diff_cc_minus_cr",
            ],
        )
        writer.writeheader()
        writer.writerows(purple_compare_rows)

    with (OUTPUT_DIR / "disagreement_blank_summary_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "session_id", "color", "task", "total_rows", "blank_rows", "blank_rate"],
        )
        writer.writeheader()
        writer.writerows(disagreement_rows)

    with (OUTPUT_DIR / "disagreement_blank_neighbor_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "color", "task", "prev_label", "next_label", "blank_count"],
        )
        writer.writeheader()
        for (split, color, task, prev_label, next_label), count in sorted(disagreement_neighbor_counts.items()):
            writer.writerow(
                {
                    "split": split,
                    "color": color,
                    "task": task,
                    "prev_label": prev_label,
                    "next_label": next_label,
                    "blank_count": str(count),
                }
            )

    with (OUTPUT_DIR / "nan_summary_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "session_id", "color", "task", "total_rows", "nan_rows", "nan_rate"],
        )
        writer.writeheader()
        writer.writerows(nan_rows)

    print(f"Sessions: train-cc={len(split_to_sessions['train-cc'])}, train-cr={len(split_to_sessions['train-cr'])}")
    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
