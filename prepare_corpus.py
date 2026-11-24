# Copyright 2026 Nuri ML contributors
"""Extract a reproducible, category-balanced file sample for fastText pretraining."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import typer
from pydantic import BaseModel, ConfigDict

from nuri_data import DEFAULT_DATA_DIR, normalize_syllables, read_splits


class Sentence(BaseModel):
    """Read only the text, never named-entity annotations."""

    model_config = ConfigDict(frozen=True)
    sentence: str


class Article(BaseModel):
    """The labeled corpus provides already segmented sentences."""

    model_config = ConfigDict(frozen=True)
    title: tuple[Sentence, ...]
    subtitle: tuple[Sentence, ...]
    content: tuple[Sentence, ...]


class CorpusFile(BaseModel):
    """Parse the relevant fields at the ZIP/JSON boundary."""

    model_config = ConfigDict(frozen=True)
    named_entity: tuple[Article, ...]


def clean_text(text: str) -> str:
    """Keep fastText's reserved label syntax out of ordinary input."""
    return normalize_syllables(text).replace("__label__", "label")


def extract_sentences(payload: bytes) -> tuple[str, ...]:
    """Parse labeled-corpus sentences without exposing annotations as features."""
    document = CorpusFile.model_validate_json(payload)
    return tuple(
        clean_text(row.sentence)
        for article in document.named_entity
        for row in (*article.title, *article.subtitle, *article.content)
        if clean_text(row.sentence)
    )


def main(
    *,
    corpus_dir: Annotated[Path, typer.Option(help="Root containing Training/TL1.zip")] = Path(
        "data"
    ),
    output_dir: Annotated[Path, typer.Option(help="Generated corpus and manifest")] = Path(
        "artifacts/corpus"
    ),
    files_per_category: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """Sample files by seeded hash within each category; exclude Nuri split texts."""
    archives = sorted(corpus_dir.rglob("TL1.zip"))
    if len(archives) != 1:
        reason = "Expected exactly one training TL1.zip"
        raise typer.BadParameter(reason)
    archive = archives[0]
    train, val = read_splits(DEFAULT_DATA_DIR)
    excluded = {clean_text(row.text) for row in (*train, *val)}
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    duplicates = 0
    overlaps = 0
    with ZipFile(archive) as zipped:
        categories: defaultdict[str, list[str]] = defaultdict(list)
        for member in zipped.namelist():
            if member.endswith(".json"):
                categories[member.split("/")[0]].append(member)
        with (output_dir / "sentences.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for category, members in sorted(categories.items()):
                members.sort(key=lambda name: hashlib.sha256(f"42:{name}".encode()).hexdigest())
                for member in members[:files_per_category]:
                    selected.append(member)
                    for sentence in extract_sentences(zipped.read(member)):
                        if sentence in excluded:
                            overlaps += 1
                            continue
                        if sentence in seen:
                            duplicates += 1
                            continue
                        seen.add(sentence)
                        counts[category] += 1
                        _ = handle.write(sentence + "\n")
    with archive.open("rb") as archive_handle:
        archive_hash = hashlib.file_digest(archive_handle, "sha256").hexdigest()
    manifest = {
        "archive": archive.as_posix(),
        "archive_sha256": archive_hash,
        "sampling": "first N SHA256('42:'+member) files per archive category",
        "files_per_category": files_per_category,
        "selected_members": selected,
        "sentences_by_category": dict(counts),
        "sentences": sum(counts.values()),
        "removed_duplicates": duplicates,
        "removed_nuri_exact_matches": overlaps,
        "exclusion": "normalized full-text matches against both Nuri splits; not fuzzy overlap",
        "corpus_sha256": hashlib.sha256((output_dir / "sentences.txt").read_bytes()).hexdigest(),
    }
    _ = (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Prepared {sum(counts.values())} sentences from {len(selected)} files")


if __name__ == "__main__":
    typer.run(main)
