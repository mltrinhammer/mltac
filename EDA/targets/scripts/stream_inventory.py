from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path("X:/")
PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"


INFO_RE = re.compile(r'<info[^>]*sr="([^"]+)"[^>]*dim="([^"]+)"[^>]*')
CHUNK_RE = re.compile(r'<chunk[^>]*num="([^"]+)"[^>]*')


def parse_stream_header(path: Path) -> tuple[str, str, str]:
    sr = ""
    dim = ""
    num = ""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return sr, dim, num
    info = INFO_RE.search(txt)
    chunk = CHUNK_RE.search(txt)
    if info:
        sr = info.group(1)
        dim = info.group(2)
    if chunk:
        num = chunk.group(1)
    return sr, dim, num


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    summary: dict[tuple[str, str], int] = defaultdict(int)

    for split in ("train-cc", "train-cr", "val-cc", "val-cr", "test-cc", "test-cr"):
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            continue
        sessions = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for session in sessions:
            headers = sorted(session.glob("*.stream"))
            for header in headers:
                stem = header.name[:-len(".stream")]
                stream_bin = session / f"{stem}.stream~"
                entity = stem.split(".", 1)[0] if "." in stem else "unknown"
                feature = stem.split(".", 1)[1] if "." in stem else stem
                sr, dim, num = parse_stream_header(header)
                rows.append(
                    {
                        "split": split,
                        "session_id": session.name,
                        "entity": entity,
                        "feature": feature,
                        "stream_header_file": header.name,
                        "has_binary_stream": "1" if stream_bin.exists() else "0",
                        "sr": sr,
                        "dim": dim,
                        "num": num,
                    }
                )
                summary[(split, feature)] += 1

    with (OUTPUT_DIR / "stream_inventory_by_session.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "session_id",
                "entity",
                "feature",
                "stream_header_file",
                "has_binary_stream",
                "sr",
                "dim",
                "num",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "stream_coverage_by_split_feature.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "feature", "session_count_with_feature"])
        writer.writeheader()
        for (split, feature), count in sorted(summary.items()):
            writer.writerow({"split": split, "feature": feature, "session_count_with_feature": str(count)})

    print(f"Wrote stream inventory outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()

