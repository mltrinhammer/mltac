"""Speech-turn boundary computation from transcript annotations.

The NoXi dataset provides per-role transcript files where each row contains
the start and end time (in seconds) of a speech utterance.  This module reads
those files and partitions a session timeline into *turns* — non-overlapping
segments that run from one speaker's onset to the next speaker's onset.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TurnSegment:
    """One speech turn within a dyadic session.

    A turn runs from when one speaker begins talking until the other speaker
    begins talking (or until the session ends).  The ``speaker`` field records
    who initiated the turn.
    """

    speaker: str       # "novice" or "expert"
    start_frame: int   # inclusive, at *rate* Hz
    end_frame: int     # exclusive, at *rate* Hz


def read_transcript(path: Path) -> list[tuple[float, float]]:
    """Read a two-column transcript annotation CSV.

    Expected format — one row per utterance, two columns (start_sec, end_sec).
    An optional header line is auto-detected via :func:`csv.Sniffer`.  Returns
    a sorted list of ``(start_sec, end_sec)`` tuples.
    """

    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        # Auto-detect delimiter (common options: semicolon, comma, tab).
        delimiter = ";"
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
            delimiter = dialect.delimiter
        except csv.Error:
            pass

        has_header = False
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            pass

        reader = csv.reader(handle, delimiter=delimiter)
        if has_header:
            next(reader, None)

        for line in reader:
            if not line or len(line) < 2:
                continue
            try:
                start = float(line[0].strip())
                end = float(line[1].strip())
            except (ValueError, IndexError):
                continue
            if end > start:
                rows.append((start, end))

    rows.sort(key=lambda t: t[0])
    return rows


def compute_turn_segments(
    novice_transcript: list[tuple[float, float]],
    expert_transcript: list[tuple[float, float]],
    session_len_frames: int,
    rate: float = 25.0,
) -> list[TurnSegment]:
    """Partition a session into non-overlapping speaker turns.

    Algorithm
    ---------
    1. Collect every speech **onset** from both roles.
    2. Sort chronologically.
    3. Each segment spans from one onset to the next (or session end).
    4. Assign each segment to the role whose onset initiated it.
    5. Convert seconds → frame indices, clamp to ``[0, session_len_frames]``.
    6. Drop zero-length segments.
    """

    onsets: list[tuple[float, str]] = []
    for start, _end in novice_transcript:
        onsets.append((start, "novice"))
    for start, _end in expert_transcript:
        onsets.append((start, "expert"))

    if not onsets:
        return []

    # Stable sort: when two onsets share the same time, the role that appeared
    # first in the list (novice before expert) keeps its position.
    onsets.sort(key=lambda t: t[0])

    segments: list[TurnSegment] = []
    session_end_sec = session_len_frames / rate

    for idx, (onset_sec, role) in enumerate(onsets):
        if idx + 1 < len(onsets):
            next_sec = onsets[idx + 1][0]
        else:
            next_sec = session_end_sec

        start_frame = max(0, round(onset_sec * rate))
        end_frame = min(session_len_frames, round(next_sec * rate))

        if end_frame > start_frame:
            segments.append(TurnSegment(speaker=role, start_frame=start_frame, end_frame=end_frame))

    return segments
