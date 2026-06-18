from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.turns import compute_turn_segments, read_transcript


ROLE_ORDER = ("novice", "expert")
TRANSCRIPT_SUFFIX = "audio.transcript.annotation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired turn manifest from a transformed role-level manifest.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--min-turn-frames", type=int, default=1)
    return parser.parse_args()


def infer_branch_name(rows: list[dict[str, str]]) -> str:
    feature_sets = sorted({row.get("feature_set", "features") for row in rows})
    suffixes = sorted({row.get("transform_suffix", row.get("transform_method", "raw")) for row in rows})
    feature_set = feature_sets[0] if len(feature_sets) == 1 else "features"
    suffix = suffixes[0] if len(suffixes) == 1 else "mixed"
    return f"{feature_set}_{suffix}_turns"


def default_manifest_path(branch_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{branch_name}.csv"


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        role = row.get("role")
        if role not in ROLE_ORDER:
            continue
        grouped[(row["dataset"], row["session_id"])] [role] = row
    return dict(grouped)


def locate_transcript(transcript_root: Path, dataset: str, session_id: str, role: str) -> Path | None:
    base = transcript_root / dataset
    if not base.exists():
        return None
    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        candidate = split_dir / session_id / f"{role}.{TRANSCRIPT_SUFFIX}"
        if candidate.exists():
            return candidate
    return None


def build_turn_rows(
    dataset: str,
    session_id: str,
    role_rows: dict[str, dict[str, str]],
    transcript_root: Path,
    min_turn_frames: int,
) -> list[dict[str, object]]:
    missing_roles = [role for role in ROLE_ORDER if role not in role_rows]
    if missing_roles:
        raise ValueError(f"Missing roles for {dataset}/{session_id}: {missing_roles}")

    novice_row = role_rows["novice"]
    expert_row = role_rows["expert"]
    if novice_row["model_split"] != expert_row["model_split"]:
        raise ValueError(
            f"Split mismatch for {dataset}/{session_id}: novice={novice_row['model_split']}, expert={expert_row['model_split']}"
        )
    if novice_row["n_features"] != expert_row["n_features"]:
        raise ValueError(
            f"Feature dimension mismatch for {dataset}/{session_id}: novice={novice_row['n_features']}, expert={expert_row['n_features']}"
        )

    novice_transcript_path = locate_transcript(transcript_root, dataset, session_id, "novice")
    expert_transcript_path = locate_transcript(transcript_root, dataset, session_id, "expert")
    if novice_transcript_path is None or expert_transcript_path is None:
        return []

    novice_transcript = read_transcript(novice_transcript_path)
    expert_transcript = read_transcript(expert_transcript_path)
    session_len = min(int(novice_row["aligned_len"]), int(expert_row["aligned_len"]))
    segments = compute_turn_segments(novice_transcript, expert_transcript, session_len)

    rows: list[dict[str, object]] = []
    for turn_idx, segment in enumerate(segments):
        turn_len = segment.end_frame - segment.start_frame
        if turn_len < min_turn_frames:
            continue
        rows.append(
            {
                "dataset": dataset,
                "session_id": session_id,
                "model_split": novice_row["model_split"],
                "feature_set": novice_row.get("feature_set", ""),
                "transform_method": novice_row.get("transform_method", ""),
                "transform_scope": novice_row.get("transform_scope", "shared"),
                "transform_suffix": novice_row.get("transform_suffix", novice_row.get("transform_method", "raw")),
                "turn_idx": turn_idx,
                "speaker": segment.speaker,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "turn_len": turn_len,
                "session_aligned_len": session_len,
                "novice_tensor_relative_path": novice_row["tensor_relative_path"],
                "expert_tensor_relative_path": expert_row["tensor_relative_path"],
                "novice_aligned_len": novice_row["aligned_len"],
                "expert_aligned_len": expert_row["aligned_len"],
                "n_features_per_role": novice_row["n_features"],
                "metadata_json": json.dumps(
                    {
                        "novice_transcript": str(novice_transcript_path.relative_to(transcript_root)).replace("\\", "/"),
                        "expert_transcript": str(expert_transcript_path.relative_to(transcript_root)).replace("\\", "/"),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {args.input_manifest}")

    branch_name = infer_branch_name(rows)
    output_manifest = args.output_manifest or default_manifest_path(branch_name)

    processed_rows: list[dict[str, object]] = []
    skipped_sessions = 0
    for (dataset, session_id), role_rows in sorted(group_rows(rows).items()):
        turn_rows = build_turn_rows(
            dataset=dataset,
            session_id=session_id,
            role_rows=role_rows,
            transcript_root=args.transcript_root,
            min_turn_frames=args.min_turn_frames,
        )
        if not turn_rows:
            skipped_sessions += 1
            continue
        processed_rows.extend(turn_rows)

    if not processed_rows:
        raise RuntimeError("No turn rows were produced.")

    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)
    print(f"Input manifest: {args.input_manifest}")
    print(f"Turn branch: {branch_name}")
    print(f"Wrote turn rows: {len(processed_rows)}")
    print(f"Skipped sessions: {skipped_sessions}")
    print(f"Output manifest: {output_manifest}")


if __name__ == "__main__":
    main()