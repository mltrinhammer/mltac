"""Inventory PinSoRo participant metadata and join it to CR-social diagnostics."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT.parents[1] / "PinSoRo"
DEFAULT_SESSION_METRICS = (
    PROJECT_ROOT
    / "model improvement/test 5 receptive field ablation/results/"
    "w2400_s1200_l5_k11_causal/deep_error_analysis/analysis/"
    "cr_social_session_diagnostics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "model improvement/test 5 receptive field ablation/results/"
    "w2400_s1200_l5_k11_causal/deep_error_analysis/metadata_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PinSoRo metadata coverage.")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--session-metrics", type=Path, default=DEFAULT_SESSION_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotation_value(archive: zipfile.ZipFile, name: str) -> str:
    text = archive.read(name).decode("utf-8-sig").strip()
    fields = text.split(";")
    return fields[2] if len(fields) >= 3 else text


def read_metadata(archive_root: Path) -> list[dict[str, object]]:
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for archive_path in sorted(archive_root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not name.endswith((".age.annotation.csv", ".gender.annotation.csv")):
                    continue
                parts = Path(name).parts
                if len(parts) != 3:
                    continue
                source_split, session_id, filename = parts
                role, field, _, _ = filename.split(".")
                key = (source_split, session_id, role)
                row = rows.setdefault(
                    key,
                    {
                        "archive": archive_path.name,
                        "source_split": source_split,
                        "domain": "CR" if source_split.endswith("-cr") else "CC",
                        "session_id": session_id,
                        "role": role,
                        "age": "",
                        "gender": "",
                    },
                )
                row[field] = annotation_value(archive, name)
    return sorted(rows.values(), key=lambda row: (row["source_split"], row["session_id"], row["role"]))


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.archive_root)
    write_rows(args.output_dir / "participant_metadata.csv", metadata)
    by_session = {
        (str(row["session_id"]), str(row["role"])): row for row in metadata
    }
    joined = []
    with args.session_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata_row = by_session.get((row["session_id"], row["role"]), {})
            joined.append(
                {
                    **row,
                    "age": metadata_row.get("age", ""),
                    "gender": metadata_row.get("gender", ""),
                    "metadata_source_split": metadata_row.get("source_split", ""),
                }
            )
    write_rows(args.output_dir / "cr_social_sessions_with_metadata.csv", joined)

    summary = []
    for field in ("age", "gender"):
        values = sorted({row[field] for row in joined if row[field] != ""}, key=float)
        for value in values:
            subset = [row for row in joined if row[field] == value]
            kappas = [float(row["kappa"]) for row in subset]
            summary.append(
                {
                    "field": field,
                    "value": value,
                    "n_sessions": len(subset),
                    "mean_session_kappa": sum(kappas) / len(kappas),
                    "n_nonpositive_kappa": sum(kappa <= 0 for kappa in kappas),
                }
            )
    write_rows(args.output_dir / "cr_social_metadata_summary.csv", summary)
    print(f"Wrote {len(metadata)} metadata rows and {len(joined)} CR-session joins")


if __name__ == "__main__":
    main()
