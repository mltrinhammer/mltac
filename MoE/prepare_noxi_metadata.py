"""Extract NOXI / NOXI-J role metadata from the raw dataset archives."""

from __future__ import annotations

import argparse
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"


ROLES = ("novice", "expert")


@dataclass(frozen=True)
class Corpus:
    name: str
    dataset: str
    data_dir: str
    zip_root: Path
    split_zips: tuple[str, ...]


CORPORA = {
    "noxi": Corpus(
        name="noxi",
        dataset="noxi",
        data_dir="noxi_data",
        zip_root=Path("/work/ACM/Noxi"),
        split_zips=("train.zip", "val.zip", "test-base.zip", "test-additional.zip"),
    ),
    "noxi_j": Corpus(
        name="noxi_j",
        dataset="noxij",
        data_dir="noxi_j_data",
        zip_root=Path("/work/ACM/NoxiJ"),
        split_zips=("train.zip", "val.zip", "test.zip"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NOXI role metadata tables from raw zip archives.")
    parser.add_argument("--corpora", nargs="+", choices=sorted(CORPORA), default=sorted(CORPORA))
    parser.add_argument("--noxi-zip-root", type=Path, default=CORPORA["noxi"].zip_root)
    parser.add_argument("--noxi-j-zip-root", type=Path, default=CORPORA["noxi_j"].zip_root)
    return parser.parse_args()


def corpus_with_args(corpus: Corpus, args: argparse.Namespace) -> Corpus:
    if corpus.name == "noxi":
        return Corpus(corpus.name, corpus.dataset, corpus.data_dir, args.noxi_zip_root, corpus.split_zips)
    if corpus.name == "noxi_j":
        return Corpus(corpus.name, corpus.dataset, corpus.data_dir, args.noxi_j_zip_root, corpus.split_zips)
    raise ValueError(corpus.name)


def read_annotation_value(archive: zipfile.ZipFile, path: str) -> str:
    try:
        text = archive.read(path).decode("utf-8-sig", errors="replace")
    except KeyError:
        return ""
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(";")]
        if len(parts) >= 3 and parts[2] != "":
            return parts[2]
        if len(parts) == 1 and parts[0] != "":
            return parts[0]
    return ""


def archive_rows(corpus: Corpus, zip_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        sessions = sorted(
            {
                parts[1]
                for name in names
                if len(parts := name.split("/")) >= 3 and parts[1] and parts[2]
            }
        )
        top_level = zip_path.stem
        for session_id in sessions:
            language = read_annotation_value(archive, f"{top_level}/{session_id}/language.annotation.csv")
            for role in ROLES:
                age = read_annotation_value(archive, f"{top_level}/{session_id}/{role}.age.annotation.csv")
                gender = read_annotation_value(archive, f"{top_level}/{session_id}/{role}.gender.annotation.csv")
                if not age and not gender and not language:
                    continue
                rows.append(
                    {
                        "dataset": corpus.dataset,
                        "corpus": corpus.name,
                        "source_split": top_level,
                        "session_id": session_id,
                        "role": role,
                        "age": age,
                        "gender": gender,
                        "language": language,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "corpus", "source_split", "session_id", "role", "age", "gender", "language"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for name in args.corpora:
        corpus = corpus_with_args(CORPORA[name], args)
        rows: list[dict[str, str]] = []
        for zip_name in corpus.split_zips:
            zip_path = corpus.zip_root / zip_name
            if not zip_path.is_file():
                raise FileNotFoundError(zip_path)
            rows.extend(archive_rows(corpus, zip_path))
        if not rows:
            raise RuntimeError(f"No metadata rows found for {corpus.name}.")
        output = MOE_ROOT / corpus.data_dir / "outputs" / "metadata" / "role_metadata.csv"
        write_csv(output, rows)
        print(f"{corpus.name}: rows={len(rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
