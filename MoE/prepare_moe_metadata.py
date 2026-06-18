"""Extract PinSoRo age/gender metadata for MoE experiments."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT.parents[1] / "PinSoRo"
DEFAULT_OUTPUT = PROJECT_ROOT / "MoE" / "moe_data" / "outputs" / "participant_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PinSoRo participant age/gender metadata.")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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
    if not args.archive_root.is_dir():
        raise FileNotFoundError(args.archive_root)
    rows = read_metadata(args.archive_root)
    if not rows:
        raise RuntimeError(f"No age/gender annotations found under {args.archive_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} metadata rows to {args.output}")


if __name__ == "__main__":
    main()
