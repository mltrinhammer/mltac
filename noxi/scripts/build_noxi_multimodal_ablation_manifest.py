from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NOXI multimodal manifest with a subset of modalities.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--combo-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(args.modalities)
    combo_name = args.combo_name.strip() or "__".join(selected)
    feature_set_combo = "+".join(selected)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.input_manifest.open("r", newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header in {args.input_manifest}")
        rows: list[dict[str, str]] = []
        for row in reader:
            available = tuple(json.loads(row["modality_order_json"]))
            missing = [name for name in selected if name not in available]
            if missing:
                raise RuntimeError(f"Missing modalities {missing} in row {row.get('dataset')}/{row.get('session_id')}")
            specs = json.loads(row["modalities_json"])
            reduced_specs = {name: specs[name] for name in selected}
            row = dict(row)
            row["combo_name"] = combo_name
            row["feature_set_combo"] = feature_set_combo
            row["modality_order_json"] = json.dumps(list(selected))
            row["modalities_json"] = json.dumps(reduced_specs, sort_keys=True)
            rows.append(row)

    with args.output_manifest.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input manifest: {args.input_manifest}")
    print(f"Output manifest: {args.output_manifest}")
    print(f"Modalities: {', '.join(selected)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
