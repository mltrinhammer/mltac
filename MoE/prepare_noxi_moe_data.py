"""Prepare separated NOXI / NOXI-J data for MoE expert experiments.

The historical NOXI preprocessing path can mix NOXI and NOXI-J through shared
manifests and normalizers. This wrapper keeps the two corpora isolated while
reusing the existing tested tensor preparation scripts.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MOE_ROOT = PROJECT_ROOT / "MoE"
DEFAULT_FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
ROLES = ("expert", "novice")
STREAMS = {
    "visual_videomae": ("videomae",),
    "audio_w2vbert2": ("audio.w2vbert2_embeddings",),
    "text_xlm_roberta": ("audio.xlm_roberta_embeddings",),
}
TARGET_SUFFIX = "engagement.annotation.csv"


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    dataset: str
    cache_dir: str
    default_zip_root: Path
    data_dir: str
    splits: dict[str, str]


@dataclass(frozen=True)
class LocalZipEntry:
    filename: str
    file_size: int
    compress_size: int
    compress_type: int
    flag_bits: int
    data_offset: int


CORPORA = {
    "noxi": CorpusConfig(
        name="noxi",
        dataset="noxi",
        cache_dir="noxi_a",
        default_zip_root=Path("/work/ACM/Noxi"),
        data_dir="noxi_data",
        splits={
            "train": "train_internal",
            "val": "val_internal",
            "test-base": "test_internal",
            "test-additional": "test_additional",
        },
    ),
    "noxi_j": CorpusConfig(
        name="noxi_j",
        dataset="noxij",
        cache_dir="noxi_b",
        default_zip_root=Path("/work/ACM/NoxiJ"),
        data_dir="noxi_j_data",
        splits={
            "train": "train_internal",
            "val": "val_internal",
            "test": "test_internal",
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected NOXI/NOXI-J modality streams, align them to the "
            "25 Hz target grid, fit train-only normalizers, and build 80s/40s "
            "dyadic window manifests."
        )
    )
    parser.add_argument("--corpora", nargs="+", choices=sorted(CORPORA), default=sorted(CORPORA))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES), choices=sorted(STREAMS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--noxi-zip-root", type=Path, default=CORPORA["noxi"].default_zip_root)
    parser.add_argument("--noxi-j-zip-root", type=Path, default=CORPORA["noxi_j"].default_zip_root)
    parser.add_argument("--output-root", type=Path, default=MOE_ROOT)
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def zip_root(args: argparse.Namespace, corpus: CorpusConfig) -> Path:
    if corpus.name == "noxi":
        return args.noxi_zip_root
    if corpus.name == "noxi_j":
        return args.noxi_j_zip_root
    raise KeyError(corpus.name)


def data_root(args: argparse.Namespace, corpus: CorpusConfig) -> Path:
    return args.output_root / corpus.data_dir


def raw_manifest_path(root: Path) -> Path:
    return root / "outputs" / "manifests" / "model_raw_manifest_train_with_split.csv"


def stream_manifest_path(root: Path) -> Path:
    return root / "outputs" / "manifests" / "model_raw_manifest_streams_train.csv"


def selected_stream_names(features: list[str]) -> set[str]:
    names: set[str] = set()
    for feature in features:
        names.update(STREAMS[feature])
    return names


def is_selected_member(member: str, stream_names: set[str]) -> bool:
    if member.endswith("/"):
        return False
    filename = Path(member).name
    if filename.endswith(f".{TARGET_SUFFIX}"):
        return True
    for stream in stream_names:
        if filename.endswith(f".{stream}.stream") or filename.endswith(f".{stream}.stream~"):
            return True
    return False


def split_name_from_zip(zip_path: Path) -> str:
    return zip_path.stem


def split_zips(corpus: CorpusConfig, root: Path) -> list[Path]:
    paths = [root / f"{split}.zip" for split in corpus.splits]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing zip archive(s): {missing}")
    return paths


def zip64_sizes(extra: bytes, compressed_size: int, file_size: int) -> tuple[int, int]:
    """Return local-header sizes, resolving Zip64 extra fields when present."""

    offset = 0
    current_file_size = file_size
    current_compressed_size = compressed_size
    while offset + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, offset)
        offset += 4
        data = extra[offset : offset + data_size]
        offset += data_size
        if header_id != 0x0001:
            continue
        data_offset = 0
        if current_file_size == 0xFFFFFFFF and data_offset + 8 <= len(data):
            current_file_size = struct.unpack_from("<Q", data, data_offset)[0]
            data_offset += 8
        if current_compressed_size == 0xFFFFFFFF and data_offset + 8 <= len(data):
            current_compressed_size = struct.unpack_from("<Q", data, data_offset)[0]
        break
    return current_compressed_size, current_file_size


def iter_local_zip_entries(path: Path) -> list[LocalZipEntry]:
    """Scan local file headers without relying on the central directory."""

    entries: list[LocalZipEntry] = []
    with path.open("rb") as handle:
        while True:
            signature = handle.read(4)
            if not signature:
                break
            if signature in {b"PK\x01\x02", b"PK\x05\x06", b"PK\x06\x06"}:
                break
            if signature != b"PK\x03\x04":
                raise zipfile.BadZipFile(
                    f"Unexpected zip signature {signature!r} in {path} at {handle.tell() - 4}"
                )
            header = handle.read(26)
            if len(header) != 26:
                raise zipfile.BadZipFile(f"Truncated local header in {path}")
            (
                _version,
                flag_bits,
                compress_type,
                _mod_time,
                _mod_date,
                _crc,
                compressed_size,
                file_size,
                filename_len,
                extra_len,
            ) = struct.unpack("<HHHHHIIIHH", header)
            filename = handle.read(filename_len).decode("utf-8", errors="replace")
            extra = handle.read(extra_len)
            compressed_size, file_size = zip64_sizes(extra, compressed_size, file_size)
            if flag_bits & 0x08:
                raise zipfile.BadZipFile(
                    f"Unsupported data-descriptor entry in {path}: {filename}"
                )
            data_offset = handle.tell()
            entries.append(
                LocalZipEntry(
                    filename=filename,
                    file_size=file_size,
                    compress_size=compressed_size,
                    compress_type=compress_type,
                    flag_bits=flag_bits,
                    data_offset=data_offset,
                )
            )
            handle.seek(compressed_size, 1)
    return entries


def copy_entry_payload(zip_path: Path, entry: LocalZipEntry, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zip_path.open("rb") as source:
        source.seek(entry.data_offset)
        remaining = entry.compress_size
        if entry.compress_type == 0:
            with target.open("wb") as output:
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(f"Unexpected EOF while extracting {entry.filename}")
                    output.write(chunk)
                    remaining -= len(chunk)
            return
        if entry.compress_type == 8:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            with target.open("wb") as output:
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(f"Unexpected EOF while extracting {entry.filename}")
                    remaining -= len(chunk)
                    output.write(decompressor.decompress(chunk))
                output.write(decompressor.flush())
            return
    raise NotImplementedError(
        f"Unsupported compression method {entry.compress_type} for {entry.filename}"
    )


def iter_zip_entries(path: Path) -> list[LocalZipEntry]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                LocalZipEntry(
                    filename=info.filename,
                    file_size=info.file_size,
                    compress_size=info.compress_size,
                    compress_type=info.compress_type,
                    flag_bits=info.flag_bits,
                    data_offset=-1,
                )
                for info in archive.infolist()
            ]
    return iter_local_zip_entries(path)


def copy_archive_entry(zip_path: Path, entry: LocalZipEntry, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if entry.data_offset >= 0:
        copy_entry_payload(zip_path, entry, target)
        return
    with zipfile.ZipFile(zip_path) as archive, archive.open(entry.filename) as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def extract_selected(
    corpus: CorpusConfig,
    archive_root: Path,
    cache_root: Path,
    features: list[str],
    force: bool,
    dry_run: bool,
) -> None:
    stream_names = selected_stream_names(features)
    extract_root = cache_root / corpus.cache_dir
    for zip_path in split_zips(corpus, archive_root):
        try:
            entries = iter_zip_entries(zip_path)
        except zipfile.BadZipFile as exc:
            print(f"{corpus.name}: skip_unreadable_archive {zip_path}: {exc}", flush=True)
            continue
        selected = [entry for entry in entries if is_selected_member(entry.filename, stream_names)]
        total_gib = sum(entry.file_size for entry in selected) / (1024**3)
        print(
            f"{corpus.name}: {zip_path.name} selected_files={len(selected)} "
            f"uncompressed_gib={total_gib:.2f}",
            flush=True,
        )
        if dry_run:
            continue
        for entry in selected:
            target = extract_root / entry.filename
            if target.is_file() and target.stat().st_size == entry.file_size and not force:
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            copy_archive_entry(zip_path, entry, temporary)
            shutil.move(str(temporary), str(target))


def build_raw_manifests(
    corpus: CorpusConfig,
    archive_root: Path,
    root: Path,
    features: list[str],
    force: bool,
    dry_run: bool,
) -> None:
    manifest_path = raw_manifest_path(root)
    streams_path = stream_manifest_path(root)
    if manifest_path.is_file() and streams_path.is_file() and not force:
        print(f"{corpus.name}: skip_existing {manifest_path}", flush=True)
        return

    stream_names = selected_stream_names(features)
    manifest_rows: list[dict[str, str]] = []
    stream_rows: list[dict[str, str]] = []
    seen_role_rows: set[tuple[str, str, str]] = set()

    for zip_path in split_zips(corpus, archive_root):
        split = split_name_from_zip(zip_path)
        model_split = corpus.splits[split]
        by_session: dict[str, set[str]] = {}
        targets: set[tuple[str, str]] = set()
        stream_headers: set[tuple[str, str, str]] = set()
        stream_binaries: set[tuple[str, str, str]] = set()

        try:
            entries = iter_zip_entries(zip_path)
        except zipfile.BadZipFile as exc:
            print(f"{corpus.name}: skip_unreadable_archive {zip_path}: {exc}", flush=True)
            continue
        for entry in entries:
            parts = Path(entry.filename).parts
            if len(parts) != 3 or parts[0] != split:
                continue
            _, session_id, filename = parts
            if "." not in filename:
                continue
            role = filename.split(".", 1)[0]
            if role not in ROLES:
                continue
            by_session.setdefault(session_id, set()).add(role)
            if filename.endswith(f".{TARGET_SUFFIX}"):
                targets.add((session_id, role))
                continue
            for stream in stream_names:
                if filename == f"{role}.{stream}.stream":
                    stream_headers.add((session_id, role, stream))
                elif filename == f"{role}.{stream}.stream~":
                    stream_binaries.add((session_id, role, stream))

        for session_id in sorted(by_session):
            for role in ROLES:
                if role not in by_session[session_id]:
                    continue
                key = (split, session_id, role)
                if key not in seen_role_rows:
                    target_relative = (
                        f"{split}/{session_id}/{role}.{TARGET_SUFFIX}"
                        if (session_id, role) in targets
                        else ""
                    )
                    manifest_rows.append(
                        {
                            "dataset": corpus.dataset,
                            "session_id": session_id,
                            "role": role,
                            "model_split": model_split,
                            "target_relative_path": target_relative,
                        }
                    )
                    seen_role_rows.add(key)
                for stream in sorted(stream_names):
                    if (session_id, role, stream) not in stream_binaries:
                        continue
                    stream_rows.append(
                        {
                            "dataset": corpus.dataset,
                            "session_id": session_id,
                            "role": role,
                            "stream_name": stream,
                            "local_relative_path": f"{split}/{session_id}/{role}.{stream}.stream",
                            "binary_local_relative_path": f"{split}/{session_id}/{role}.{stream}.stream~",
                            "has_header": "yes"
                            if (session_id, role, stream) in stream_headers
                            else "no",
                        }
                    )

    print(
        f"{corpus.name}: raw_role_rows={len(manifest_rows)} stream_rows={len(stream_rows)}",
        flush=True,
    )
    if dry_run:
        print(f"{corpus.name}: would_write {manifest_path}", flush=True)
        print(f"{corpus.name}: would_write {streams_path}", flush=True)
        return

    from src.acm_pipeline.io import write_csv

    write_csv(
        manifest_path,
        ["dataset", "session_id", "role", "model_split", "target_relative_path"],
        manifest_rows,
    )
    write_csv(
        streams_path,
        [
            "dataset",
            "session_id",
            "role",
            "stream_name",
            "local_relative_path",
            "binary_local_relative_path",
            "has_header",
        ],
        stream_rows,
    )


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def maybe_run(command: list[str], output: Path, force: bool, dry_run: bool) -> None:
    if output.is_file() and output.stat().st_size > 0 and not force:
        print(f"skip_existing {output}", flush=True)
        return
    run(command, dry_run)


def align_manifest(root: Path, feature: str) -> Path:
    return root / "outputs" / "manifests" / f"{feature}_25hz.csv"


def norm_manifest(root: Path, feature: str) -> Path:
    return root / "outputs" / "manifests" / f"{feature}_raw.csv"


def window_manifest(root: Path, feature: str, window_size: int, stride: int) -> Path:
    return (
        root
        / "outputs"
        / f"windows_w{window_size}_s{stride}"
        / f"{feature}_w{window_size}_s{stride}_dyadic.csv"
    )


def preprocess_corpus(args: argparse.Namespace, corpus: CorpusConfig) -> None:
    archive_root = zip_root(args, corpus)
    root = data_root(args, corpus)
    cache_root = root / "cache"
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        extract_selected(corpus, archive_root, cache_root, args.features, args.force, args.dry_run)
    build_raw_manifests(corpus, archive_root, root, args.features, args.force, args.dry_run)

    if not args.skip_align:
        for feature in args.features:
            output = align_manifest(root, feature)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/noxi_prepare_feature_tensors_25hz.py"),
                "--feature-set",
                feature,
                "--cache-root",
                str(cache_root),
                "--manifest",
                str(raw_manifest_path(root)),
                "--streams",
                str(stream_manifest_path(root)),
                "--out-root",
                str(root / "processed" / "25hz" / feature),
                "--processed-manifest",
                str(output),
                "--status-out",
                str(root / "outputs" / "status" / f"{feature}_25hz_status.csv"),
            ]
            maybe_run(command, output, args.force, args.dry_run)

    if not args.skip_normalize:
        for feature in args.features:
            output = norm_manifest(root, feature)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/noxi_fit_apply_feature_transform.py"),
                "--input-manifest",
                str(align_manifest(root, feature)),
                "--method",
                "raw",
                "--train-split",
                "train_internal",
                "--out-root",
                str(root / "processed" / "normalized" / feature),
                "--output-manifest",
                str(output),
                "--transform-dir",
                str(root / "outputs" / "transforms" / feature),
            ]
            maybe_run(command, output, args.force, args.dry_run)

    if not args.skip_windows:
        for feature in args.features:
            output = window_manifest(root, feature, args.window_size, args.stride)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/noxi_build_window_manifest.py"),
                "--input-manifest",
                str(norm_manifest(root, feature)),
                "--output-manifest",
                str(output),
                "--window-size",
                str(args.window_size),
                "--stride",
                str(args.stride),
            ]
            maybe_run(command, output, args.force, args.dry_run)


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.features) - set(STREAMS))
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")
    for corpus_name in args.corpora:
        preprocess_corpus(args, CORPORA[corpus_name])


if __name__ == "__main__":
    main()
