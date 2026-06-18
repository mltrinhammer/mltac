"""Build paired turn manifests for all C(N,2) participant pairs per session.

For datasets with more than two participants (e.g. MPII Group Interaction with
subjectPos1-4), this script creates one set of dyadic turns for every unordered
pair of participants.  Each pair gets a *composite* session ID so downstream
code (multimodal builder, model reconstruction) treats each pair as a separate
session.  A post-inference aggregation step in ``infer_tcn_multimodal.py``
recombines per-pair predictions into per-participant outputs.

Composite session ID format::

    {session_id}__pair__{role_a}__{role_b}

Usage::

    python scripts/build_allpairs_turn_manifest.py \\
        --input-manifest outputs/mpiii_eval/manifests/model_processed_manifest_audio_w2vbert2_raw.csv \\
        --transcript-root /home/mlut/mltac \\
        --output-manifest outputs/mpiii_eval/manifests/model_processed_manifest_audio_w2vbert2_raw_turns.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.turns import compute_turn_segments, read_transcript


PAIR_SEPARATOR = "__pair__"
TRANSCRIPT_SUFFIX = "audio.transcript.annotation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-pairs paired turn manifests for N-participant sessions."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--pair-separator", default=PAIR_SEPARATOR)
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


def group_rows_by_session(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """Group manifest rows by (dataset, session_id) → {role: row}."""
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        role = row.get("role", "")
        if not role:
            continue
        grouped[(row["dataset"], row["session_id"])][role] = row
    return dict(grouped)


def locate_transcript(
    transcript_root: Path, dataset: str, session_id: str, role: str,
) -> Path | None:
    """Find a transcript file by searching split subdirectories.

    Also checks a flat layout where sessions live directly in the dataset
    root (e.g. ``mpiigroupinteraction/001/...``) without a split subdir.
    """
    base = transcript_root / dataset
    if not base.exists():
        return None
    # Flat layout: sessions directly in dataset root.
    flat_candidate = base / session_id / f"{role}.{TRANSCRIPT_SUFFIX}"
    if flat_candidate.exists():
        return flat_candidate
    # Hierarchical layout: split subdirectory between dataset and session.
    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        candidate = split_dir / session_id / f"{role}.{TRANSCRIPT_SUFFIX}"
        if candidate.exists():
            return candidate
    return None


def build_pair_turn_rows(
    dataset: str,
    session_id: str,
    role_a: str,
    role_b: str,
    row_a: dict[str, str],
    row_b: dict[str, str],
    transcript_root: Path,
    pair_separator: str,
    min_turn_frames: int,
) -> list[dict[str, object]]:
    """Build turn rows for one (role_a, role_b) pair.

    role_a is placed in the "novice" model slot, role_b in the "expert" slot.
    """
    composite_id = f"{session_id}{pair_separator}{role_a}{pair_separator}{role_b}"

    transcript_a = locate_transcript(transcript_root, dataset, session_id, role_a)
    transcript_b = locate_transcript(transcript_root, dataset, session_id, role_b)
    if transcript_a is None or transcript_b is None:
        return []

    novice_transcript = read_transcript(transcript_a)
    expert_transcript = read_transcript(transcript_b)
    session_len = min(int(row_a["aligned_len"]), int(row_b["aligned_len"]))
    segments = compute_turn_segments(novice_transcript, expert_transcript, session_len)

    rows: list[dict[str, object]] = []
    for turn_idx, segment in enumerate(segments):
        turn_len = segment.end_frame - segment.start_frame
        if turn_len < min_turn_frames:
            continue
        rows.append(
            {
                "dataset": dataset,
                "session_id": composite_id,
                "model_split": row_a["model_split"],
                "feature_set": row_a.get("feature_set", ""),
                "transform_method": row_a.get("transform_method", ""),
                "transform_scope": row_a.get("transform_scope", "shared"),
                "transform_suffix": row_a.get("transform_suffix", row_a.get("transform_method", "raw")),
                "turn_idx": turn_idx,
                "speaker": segment.speaker,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "turn_len": turn_len,
                "session_aligned_len": session_len,
                "novice_tensor_relative_path": row_a["tensor_relative_path"],
                "expert_tensor_relative_path": row_b["tensor_relative_path"],
                "novice_aligned_len": row_a["aligned_len"],
                "expert_aligned_len": row_b["aligned_len"],
                "n_features_per_role": row_a["n_features"],
                "metadata_json": json.dumps(
                    {
                        "novice_role": role_a,
                        "expert_role": role_b,
                        "real_session_id": session_id,
                        "novice_transcript": str(transcript_a.relative_to(transcript_root)).replace("\\", "/"),
                        "expert_transcript": str(transcript_b.relative_to(transcript_root)).replace("\\", "/"),
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

    session_groups = group_rows_by_session(rows)
    processed_rows: list[dict[str, object]] = []
    skipped_pairs = 0

    for (dataset, session_id), role_rows in sorted(session_groups.items()):
        roles = sorted(role_rows.keys())
        if len(roles) < 2:
            print(f"  skip {dataset}/{session_id}: only {len(roles)} role(s)")
            continue

        for role_a, role_b in combinations(roles, 2):
            pair_turns = build_pair_turn_rows(
                dataset=dataset,
                session_id=session_id,
                role_a=role_a,
                role_b=role_b,
                row_a=role_rows[role_a],
                row_b=role_rows[role_b],
                transcript_root=args.transcript_root,
                pair_separator=args.pair_separator,
                min_turn_frames=args.min_turn_frames,
            )
            if not pair_turns:
                skipped_pairs += 1
                continue
            processed_rows.extend(pair_turns)

    if not processed_rows:
        raise RuntimeError("No turn rows were produced from any pair.")

    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)

    n_sessions = len(session_groups)
    n_pairs = sum(1 for roles in session_groups.values() for _ in combinations(sorted(roles.keys()), 2))
    print(f"Input manifest: {args.input_manifest}")
    print(f"Sessions: {n_sessions}  Total pairs: {n_pairs}  Skipped pairs: {skipped_pairs}")
    print(f"Wrote turn rows: {len(processed_rows)}")
    print(f"Output manifest: {output_manifest}")


if __name__ == "__main__":
    main()
