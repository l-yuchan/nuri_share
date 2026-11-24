# Copyright 2026 Nuri ML contributors
"""Compare observation-to-indicator linking models on the supplied Nuri split.

The task ranks 59 candidate indicator definitions for an observation. Train and
validation indicators are disjoint, so every validation number measures linking
to definitions absent from training.

Conditions
----------
rule_char2        fixed char-bigram cosine overlap, no learning (non-ML baseline)
tfidf_pair        char TF-IDF pair classifier fitted on training indicators
cc_zeroshot       mean-pooled cc.ko.300 cosine, no learning
e5_zeroshot       frozen multilingual-e5-small cosine, no learning
random_frozen     random token vectors, frozen, trained projection
random_train      random token vectors, trained
cc_frozen         cc.ko.300 token vectors, frozen, trained projection
cc_train          cc.ko.300 token vectors, trained
shuffled_frozen   cc.ko.300 vectors with word-vector association permuted, frozen
shuffled_train    the same permutation, trained
e5_proj           frozen e5 sentence vectors, trained projection

Run with ``uv run --extra semantic python indicator_linking_experiment.py``.
"""

import csv
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Final

import numpy as np
import torch
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from indicator_task import (
    LinkingData,
    load_linking_data,
    paired_bootstrap,
    per_row_reciprocal_rank,
    ranking_metrics,
    target_indices,
)
from nuri_data import DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from scipy import sparse
    from torch import Tensor

SEEDS: Final = (13, 42, 73, 101, 137)
CV_FOLDS: Final = 5
EPOCHS: Final = 80
BATCH_SIZE: Final = 64
PROJECTION_DIM: Final = 128
LEARNING_RATE: Final = 2e-3
LOGIT_SCALE: Final = 10.0
MAX_TOKENS: Final = 64
E5_MODEL: Final = "intfloat/multilingual-e5-small"
E5_REVISION: Final = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SHUFFLE_SEED: Final = 20261121
DEFAULT_OUTPUT_DIR: Final = Path("results_indicator_linking")
DEFAULT_CC: Final = Path("artifacts/cc.ko.300.bin")
TRAINED_STATIC: Final = (
    "random_frozen",
    "random_train",
    "cc_frozen",
    "cc_train",
    "shuffled_frozen",
    "shuffled_train",
)
ZEROSHOT_STATIC: Final = ("cc_zeroshot", "cc_maxsim")
DETERMINISTIC: Final = (
    "rule_char2",
    "tfidf_cosine",
    "tfidf_pair",
    "cc_zeroshot",
    "e5_zeroshot",
)
CONTRASTS: Final = (
    ("tfidf_pair", "tfidf_cosine"),
    ("cc_maxsim", "cc_zeroshot"),
    ("cc_maxsim", "tfidf_cosine"),
    ("cc_train", "random_train"),
    ("cc_frozen", "random_frozen"),
    ("cc_frozen", "shuffled_frozen"),
    ("cc_train", "shuffled_train"),
    ("cc_train", "cc_frozen"),
    ("e5_proj", "random_train"),
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """One fitted (or fixed) condition on one protocol and seed."""

    condition: str
    protocol: str
    seed: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    n: int


@dataclass(slots=True)
class TextBank:
    """Token ids for every text plus the indicator-definition index."""

    token_ids: "Tensor"
    mask: "Tensor"
    definition_index: "Tensor"
    row_index: "Tensor"
    vocabulary: tuple[str, ...]
    train_oov_rate: float


@dataclass(slots=True)
class Scores:
    """Protocol score matrices for one condition: cv rows and val rows."""

    cv: dict[int, np.ndarray]
    val: dict[int, np.ndarray]


def whitespace_tokens(text: str) -> list[str]:
    """Split normalized Korean text on whitespace; no morphological analyzer."""
    return [token for token in text.split(" ") if token]


def build_text_bank(data: LinkingData) -> TextBank:
    """Index every observation and definition into a fixed token vocabulary.

    The vocabulary is built from training observations and all candidate
    definitions, never from validation observations, so a validation-only word
    falls back to the shared unknown row.
    """
    rows = data.rows
    definitions = [item.definition for item in data.all_indicators]
    counts: Counter[str] = Counter()
    for row in data.train_rows:
        counts.update(whitespace_tokens(row.text))
    for definition in definitions:
        counts.update(whitespace_tokens(definition))
    vocabulary = ("<pad>", *(token for token, _ in counts.most_common()))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}

    def encode(texts: list[str]) -> tuple["Tensor", "Tensor", float]:
        ids = np.zeros((len(texts), MAX_TOKENS), dtype=np.int64)
        missing = 0
        total = 0
        for position, text in enumerate(texts):
            tokens = whitespace_tokens(text)[:MAX_TOKENS]
            total += len(tokens)
            for offset, token in enumerate(tokens):
                index = token_to_id.get(token, 0)
                missing += int(index == 0)
                ids[position, offset] = index
        lengths = np.maximum((ids != 0).sum(axis=1), 1)
        mask = (np.arange(MAX_TOKENS)[None, :] < lengths[:, None]).astype(np.float32)
        return torch.from_numpy(ids), torch.from_numpy(mask), missing / max(total, 1)

    row_ids, row_mask, train_oov = encode([row.text for row in rows])
    definition_ids, definition_mask, _ = encode(definitions)
    return TextBank(
        token_ids=torch.cat([row_ids, definition_ids], dim=0),
        mask=torch.cat([row_mask, definition_mask], dim=0),
        definition_index=torch.arange(len(rows), len(rows) + len(definitions), dtype=torch.int64),
        row_index=torch.arange(len(rows), dtype=torch.int64),
        vocabulary=vocabulary,
        train_oov_rate=float(train_oov),
    )


def load_token_vectors(
    path: Path, vocabulary: tuple[str, ...], cache_dir: Path | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """Export one vector per vocabulary entry from a fastText binary.

    The extracted table is cached by vocabulary hash, so a rerun never has to
    hold the 7 GB binary in memory again.
    """
    file_bytes = path.stat().st_size
    cache: Path | None = None
    if cache_dir is not None:
        key = hashlib.sha256("\n".join(vocabulary).encode()).hexdigest()[:16]
        cache = cache_dir / f"vocab-{key}.npz"
        if cache.exists():
            payload = np.load(cache)
            if int(payload["file_bytes"]) == file_bytes:
                info = {
                    "dimension": float(payload["dimension"]),
                    "median_norm": float(payload["median_norm"]),
                }
                return payload["vectors"], info
    import fasttext  # noqa: PLC0415 - optional dependency

    model = fasttext.load_model(str(path))
    dimension = model.get_dimension()
    vectors = np.zeros((len(vocabulary), dimension), dtype=np.float32)
    for index, token in enumerate(vocabulary):
        if index != 0:
            vectors[index] = model.get_word_vector(token)
    median_norm = float(np.median(np.linalg.norm(vectors[1:], axis=1)))
    del model
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            vectors=vectors,
            dimension=np.float64(dimension),
            median_norm=np.float64(median_norm),
            file_bytes=np.int64(file_bytes),
        )
    return vectors, {"dimension": float(dimension), "median_norm": median_norm}


def random_token_vectors(rng: np.random.Generator, rows: int, dim: int, sigma: float) -> np.ndarray:
    """Standard-normal token vectors with the padding row left at zero."""
    vectors = rng.normal(0.0, sigma, size=(rows, dim)).astype(np.float32)
    vectors[0] = 0.0
    return vectors


def shuffle_association(vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give every ordinary word another word's vector via a single seeded cycle."""
    ordinary = np.arange(1, vectors.shape[0])
    shuffled = vectors.copy()
    order = rng.permutation(ordinary)
    shuffled[ordinary] = vectors[order]
    return shuffled


class DualEncoder(torch.nn.Module):
    """Shared interface: a text index yields one L2-normalized vector."""

    def encode(self, index: "Tensor") -> "Tensor":
        """Return one vector per requested text index."""
        raise NotImplementedError


class TokenEncoder(DualEncoder):
    """Mean-pooled token embedding followed by a learned linear projection."""

    token_ids: "Tensor"
    mask: "Tensor"

    def __init__(self, bank: TextBank, vectors: np.ndarray, *, trainable: bool) -> None:
        """Index the token bank and optionally keep the embedding table trainable."""
        super().__init__()
        self.token_ids = bank.token_ids
        self.mask = bank.mask
        # Clone: a Parameter wrapping the numpy buffer would let training mutate
        # the shared table and leak folded fits into later folds, seeds and conditions.
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(np.ascontiguousarray(vectors)).clone(),
            freeze=not trainable,
            padding_idx=0,
        )
        self.projection = torch.nn.Linear(vectors.shape[1], PROJECTION_DIM, bias=False)

    def encode(self, index: "Tensor") -> "Tensor":
        """Return L2-normalized vectors for the selected texts."""
        ids = self.token_ids[index]
        mask = self.mask[index].unsqueeze(-1)
        pooled = (self.embedding(ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.nn.functional.normalize(self.projection(pooled), dim=-1)


class FixedEncoder(DualEncoder):
    """Frozen sentence vectors plus a learned linear projection."""

    table: "Tensor"

    def __init__(self, vectors: np.ndarray) -> None:
        """Hold one frozen sentence vector per text."""
        super().__init__()
        self.table = torch.from_numpy(vectors)
        self.projection = torch.nn.Linear(vectors.shape[1], PROJECTION_DIM, bias=False)

    def encode(self, index: "Tensor") -> "Tensor":
        """Return L2-normalized projected vectors for the selected texts."""
        return torch.nn.functional.normalize(self.projection(self.table[index]), dim=-1)


@dataclass(frozen=True, slots=True)
class FitSettings:
    """Optimization settings shared by every trained condition."""

    epochs: int
    learning_rate: float
    seed: int


@dataclass(frozen=True, slots=True)
class Split:
    """Row-level split used by the trained conditions."""

    fold_of_row: np.ndarray
    train_truth: np.ndarray
    val_truth: np.ndarray


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one evaluation run needs."""

    output_dir: Path
    data_dir: Path
    cc_path: Path
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float
    threads: int


def train_encoder(
    encoder: DualEncoder,
    bank: TextBank,
    *,
    train_index: "Tensor",
    targets: "Tensor",
    settings: FitSettings,
) -> None:
    """Fit the projection, and the embedding table when it is trainable."""
    torch.manual_seed(settings.seed)
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if not parameters:
        return
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    count = int(train_index.shape[0])
    for _ in range(settings.epochs):
        order = torch.randperm(count)
        for start in range(0, count, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            observation = encoder.encode(train_index[batch])
            definition = encoder.encode(bank.definition_index)
            scores = observation @ definition.transpose(0, 1) * LOGIT_SCALE
            loss = torch.nn.functional.cross_entropy(scores, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def score_rows(encoder: DualEncoder, bank: TextBank, positions: "Tensor") -> np.ndarray:
    """Return cosine scores for the selected rows against every definition."""
    observation = encoder.encode(bank.row_index[positions])
    definition = encoder.encode(bank.definition_index)
    return (observation @ definition.transpose(0, 1)).numpy()


def char_bigram_counts(texts: list[str]) -> np.ndarray:
    """Build a char bigram count matrix over a vocabulary from these texts."""
    grams: list[Counter[str]] = []
    vocabulary: dict[str, int] = {}
    for text in texts:
        counter: Counter[str] = Counter(
            text[position : position + 2] for position in range(max(len(text) - 1, 0))
        )
        counter.pop("  ", None)
        grams.append(counter)
        for gram in counter:
            vocabulary.setdefault(gram, len(vocabulary))
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row, counter in enumerate(grams):
        for gram, value in counter.items():
            matrix[row, vocabulary[gram]] = value
    return matrix


def rule_char2_scores(data: LinkingData) -> np.ndarray:
    """Score each row by fixed cosine overlap of observation and definition bigrams."""
    definitions = [item.definition for item in data.all_indicators]
    matrix = char_bigram_counts([row.text for row in data.rows] + definitions)
    observations = matrix[: len(data.rows)]
    candidates = matrix[len(data.rows) :]
    observation_norm = np.linalg.norm(observations, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    return (observations @ candidates.T) / np.maximum(observation_norm * candidate_norm.T, 1e-12)


def tfidf_cosine_scores(data: LinkingData) -> np.ndarray:
    """Score by cosine of char TF-IDF vectors, with no pair fitting.

    This isolates whether a learned pair classifier adds anything over the
    same surface representation used as a zero-shot retriever.
    """
    definitions = [item.definition for item in data.all_indicators]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=1, norm="l2"
    )
    vectorizer.fit([row.text for row in data.train_rows] + definitions)
    observation = vectorizer.transform([row.text for row in data.rows])
    candidate = vectorizer.transform(definitions)
    return np.asarray((observation @ candidate.T).todense())


def tfidf_pair_scores(data: LinkingData, *, seed: int = 42) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a char TF-IDF pair classifier; choose C inside the training indicators.

    Every (row, candidate) pair is materialized once per split, so a fold fit is
    a row subset of one sparse matrix rather than a rebuild.
    """
    from scipy import sparse  # noqa: PLC0415 - delivered with scikit-learn

    definitions = [item.definition for item in data.all_indicators]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=1, norm="l2"
    )
    vectorizer.fit([row.text for row in data.train_rows] + definitions)
    observation = vectorizer.transform([row.text for row in data.rows])
    candidate = vectorizer.transform(definitions)
    n_train = len(data.train_rows)
    n_candidates = len(definitions)
    truth = target_indices(data.train_rows, data)
    folds = indicator_folds(truth, CV_FOLDS)

    def build(row_positions: np.ndarray) -> "sparse.csr_matrix":
        # Interaction features only: additive observation or definition terms are
        # constant along one axis and cannot change the candidate ranking.
        left = observation[np.repeat(row_positions, n_candidates)]
        right = candidate[np.tile(np.arange(n_candidates), len(row_positions))]
        return sparse.csr_matrix(left.multiply(right))

    train_positions = np.arange(n_train)
    source_rows = np.repeat(train_positions, n_candidates)
    candidate_rows = np.tile(np.arange(n_candidates), n_train)
    train_pairs = build(train_positions)
    labels = (candidate_rows == truth[source_rows]).astype(np.int64)
    val_pairs = build(np.arange(n_train, len(data.rows)))

    def fit(c_value: float, rows: np.ndarray) -> LogisticRegression:
        selected = np.isin(source_rows, rows)
        # 572 positives among ~34k pairs: balanced weights keep the fit from
        # collapsing to the all-negative solution.
        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(train_pairs[selected], labels[selected])
        return model

    def score_train(model: LogisticRegression, rows: np.ndarray) -> np.ndarray:
        selected = np.isin(source_rows, rows)
        raw = model.decision_function(train_pairs[selected])
        return np.asarray(raw).reshape(len(rows), n_candidates)

    best_c, best_score = 1.0, -1.0
    for c_value in (0.1, 1.0, 10.0):
        hits = 0.0
        for fold in range(CV_FOLDS):
            model = fit(c_value, np.flatnonzero(folds != fold))
            held = np.flatnonzero(folds == fold)
            hits += float(np.sum(np.argmax(score_train(model, held), axis=1) == truth[held]))
        score_value = hits / n_train
        if score_value > best_score:
            best_c, best_score = c_value, score_value
    # Out-of-fold scores: the training rows must be scored by a model that did
    # not see their indicator, otherwise the "cv" protocol is in-sample.
    train_scores = np.zeros((n_train, n_candidates), dtype=np.float64)
    for fold in range(CV_FOLDS):
        model = fit(best_c, np.flatnonzero(folds != fold))
        held = np.flatnonzero(folds == fold)
        train_scores[held] = score_train(model, held)
    final = fit(best_c, train_positions)
    val_scores = np.asarray(final.decision_function(val_pairs)).reshape(
        len(data.rows) - n_train, n_candidates
    )
    return np.vstack([train_scores, val_scores]), {
        "c": best_c,
        "cv_recall_at_1": best_score,
    }


def indicator_folds(targets: np.ndarray, folds: int) -> np.ndarray:
    """Assign folds by indicator so held-out indicators never appear in training."""
    return targets % folds


def e5_sentence_vectors(texts: list[str], threads: int) -> np.ndarray:
    """Encode texts with the frozen multilingual-e5 model."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    torch.set_num_threads(threads)
    encoder = SentenceTransformer(E5_MODEL, revision=E5_REVISION, device="cpu")
    encoder.max_seq_length = 512
    return encoder.encode(
        ["query: " + text for text in texts],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def maxsim_scores(bank: TextBank, vectors: np.ndarray) -> np.ndarray:
    """Score by mean over definition tokens of the best matching token cosine.

    Mean pooling dilutes a short definition across a long narrative; this
    retrieval score keeps every definition token's best support, which is the
    standard way to use static word vectors for asymmetric lengths.
    """
    norm = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    table = vectors / norm
    ids = bank.token_ids.numpy()
    mask = bank.mask.numpy()
    n_rows = len(bank.row_index)
    row_tokens = ids[:n_rows]
    row_mask = mask[:n_rows]
    scores = np.zeros((n_rows, ids.shape[0] - n_rows), dtype=np.float64)
    for candidate in range(scores.shape[1]):
        definition = table[ids[n_rows + candidate]]
        definition_mask = mask[n_rows + candidate]
        similarity = np.einsum("rld,kd->rlk", table[row_tokens], definition, optimize=True)
        similarity = similarity.max(axis=1)
        similarity *= row_mask
        valid = definition_mask.sum()
        scores[:, candidate] = (similarity * definition_mask).sum(axis=1) / max(valid, 1.0)
    return scores


def zero_shot_cosine(bank: TextBank, vectors: np.ndarray) -> np.ndarray:
    """Score by cosine of mean-pooled token vectors, with no learned projection."""
    ids = bank.token_ids.numpy()
    mask = bank.mask.numpy()
    pooled = np.zeros((ids.shape[0], vectors.shape[1]), dtype=np.float64)
    for position in range(ids.shape[0]):
        length = max(int(mask[position].sum()), 1)
        pooled[position] = vectors[ids[position, :length]].mean(axis=0)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return pooled[: len(bank.row_index)] @ pooled[len(bank.row_index) :].T


def cross_validated_scores(
    factory: Callable[[], DualEncoder],
    bank: TextBank,
    split: Split,
    settings: FitSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per fold, score held-out rows, then refit on all training rows for val."""
    n_train = len(split.train_truth)
    n_val = len(split.val_truth)
    n_candidates = int(bank.definition_index.shape[0])
    cv_scores = np.zeros((n_train, n_candidates), dtype=np.float32)
    positions = torch.arange(n_train + n_val, dtype=torch.int64)
    for fold in range(CV_FOLDS):
        fit_mask = split.fold_of_row != fold
        held_mask = split.fold_of_row == fold
        torch.manual_seed(settings.seed)
        encoder = factory()
        train_encoder(
            encoder,
            bank,
            train_index=positions[:n_train][torch.from_numpy(fit_mask)],
            targets=torch.from_numpy(split.train_truth[fit_mask]),
            settings=settings,
        )
        held_rows = positions[:n_train][torch.from_numpy(held_mask)]
        cv_scores[held_mask] = score_rows(encoder, bank, held_rows)
    torch.manual_seed(settings.seed)
    encoder = factory()
    train_encoder(
        encoder,
        bank,
        train_index=positions[:n_train],
        targets=torch.from_numpy(split.train_truth),
        settings=settings,
    )
    val_scores = score_rows(encoder, bank, positions[n_train:])
    return cv_scores, val_scores


def run_deterministic(
    data: LinkingData, conditions: tuple[str, ...]
) -> tuple[dict[str, Scores], dict[str, object]]:
    """Run the rule and TF-IDF baselines, which need no seed sweep."""
    n_train = len(data.train_rows)
    output: dict[str, Scores] = {}
    metadata: dict[str, object] = {}
    if "rule_char2" in conditions:
        matrix = rule_char2_scores(data)
        output["rule_char2"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})
    if "tfidf_cosine" in conditions:
        matrix = tfidf_cosine_scores(data)
        output["tfidf_cosine"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})
    if "tfidf_pair" in conditions:
        matrix, info = tfidf_pair_scores(data)
        metadata["tfidf_pair"] = info
        output["tfidf_pair"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})
    return output, metadata


def run_static_family(
    config: RunConfig,
    bank: TextBank,
    split: Split,
    metadata: dict[str, object],
) -> dict[str, Scores]:
    """Run the mean-pooled token conditions including the shuffled controls."""
    output: dict[str, Scores] = {}
    need_vectors = any(name in config.conditions for name in (*TRAINED_STATIC, *ZEROSHOT_STATIC))
    if not need_vectors:
        return output
    if not config.cc_path.exists():
        reason = f"missing cc vectors: {config.cc_path}"
        raise FileNotFoundError(reason)
    cc_vectors, info = load_token_vectors(
        config.cc_path, bank.vocabulary, cache_dir=config.cc_path.parent
    )
    metadata["cc_ko_300"] = {
        **info,
        "sha256": sha256(config.cc_path),
        "file_bytes": config.cc_path.stat().st_size,
    }
    n_train = len(split.train_truth)
    if "cc_zeroshot" in config.conditions:
        matrix = zero_shot_cosine(bank, cc_vectors)
        output["cc_zeroshot"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})
    if "cc_maxsim" in config.conditions:
        matrix = maxsim_scores(bank, cc_vectors)
        output["cc_maxsim"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})

    rng = np.random.default_rng(SHUFFLE_SEED)
    cc_median = float(np.median(np.linalg.norm(cc_vectors[1:], axis=1)))
    random_norm = 0.1 * float(np.sqrt(cc_vectors.shape[1]))
    scale = random_norm / cc_median
    cc_scaled = cc_vectors * scale
    cc_scaled[0] = 0.0
    random_vectors = random_token_vectors(rng, cc_vectors.shape[0], cc_vectors.shape[1], 0.1)
    shuffled = shuffle_association(cc_scaled, rng)
    metadata["scaling"] = {
        "cc_median_norm": cc_median,
        "random_median_norm": float(np.median(np.linalg.norm(random_vectors[1:], axis=1))),
        "factor": scale,
        "shuffle_seed": SHUFFLE_SEED,
    }
    tables: dict[str, tuple[np.ndarray, bool]] = {
        "random_frozen": (random_vectors, False),
        "random_train": (random_vectors, True),
        "cc_frozen": (cc_scaled, False),
        "cc_train": (cc_scaled, True),
        "shuffled_frozen": (shuffled, False),
        "shuffled_train": (shuffled, True),
    }
    for name, (table, trainable) in tables.items():
        if name not in config.conditions:
            continue
        entry = Scores(cv={}, val={})
        for seed in config.seeds:
            cv, val = cross_validated_scores(
                lambda table=table, trainable=trainable: TokenEncoder(
                    bank, table, trainable=trainable
                ),
                bank,
                split,
                FitSettings(config.epochs, config.learning_rate, seed),
            )
            entry.cv[seed] = cv
            entry.val[seed] = val
        output[name] = entry
    return output


def run_contextual(
    config: RunConfig,
    data: LinkingData,
    bank: TextBank,
    split: Split,
    metadata: dict[str, object],
) -> dict[str, Scores]:
    """Run the frozen sentence-embedding conditions."""
    output: dict[str, Scores] = {}
    if not any(name in config.conditions for name in ("e5_zeroshot", "e5_proj")):
        return output
    texts = [row.text for row in data.rows] + [item.definition for item in data.all_indicators]
    vectors = e5_sentence_vectors(texts, config.threads)
    metadata["e5"] = {
        "model": E5_MODEL,
        "revision": E5_REVISION,
        "dim": int(vectors.shape[1]),
    }
    n_train = len(split.train_truth)
    n_rows = len(data.rows)
    if "e5_zeroshot" in config.conditions:
        matrix = vectors[:n_rows] @ vectors[n_rows:].T
        output["e5_zeroshot"] = Scores(cv={0: matrix[:n_train]}, val={0: matrix[n_train:]})
    if "e5_proj" in config.conditions:
        entry = Scores(cv={}, val={})
        for seed in config.seeds:
            cv, val = cross_validated_scores(
                lambda: FixedEncoder(vectors),
                bank,
                split,
                FitSettings(config.epochs, config.learning_rate, seed),
            )
            entry.cv[seed] = cv
            entry.val[seed] = val
        output["e5_proj"] = entry
    return output


def add_runs(runs: list[RunResult], name: str, entry: Scores, split: Split) -> None:
    """Append one result row per seed for both protocols."""
    for seed, matrix in entry.cv.items():
        runs.extend(summarize(name, "cv", seed, matrix, split.train_truth))
    for seed, matrix in entry.val.items():
        runs.extend(summarize(name, "val", seed, matrix, split.val_truth))


def evaluate(config: RunConfig) -> None:
    """Run every requested condition and export metrics and predictions."""
    torch.set_num_threads(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    data = load_linking_data(config.data_dir)
    bank = build_text_bank(data)
    split = Split(
        fold_of_row=indicator_folds(target_indices(data.train_rows, data), CV_FOLDS),
        train_truth=target_indices(data.train_rows, data),
        val_truth=target_indices(data.val_rows, data),
    )
    metadata: dict[str, object] = {
        "train_rows": len(data.train_rows),
        "val_rows": len(data.val_rows),
        "train_indicators": len(data.train_indicators),
        "val_indicators": len(data.val_indicators),
        "total_candidates": len(data.all_indicators),
        "indicator_overlap": len(
            {item.key for item in data.train_indicators}
            & {item.key for item in data.val_indicators}
        ),
        "vocabulary": len(bank.vocabulary),
        "train_oov_rate": bank.train_oov_rate,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "epochs": config.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": config.learning_rate,
        "projection_dim": PROJECTION_DIM,
        "logit_scale": LOGIT_SCALE,
        "cv_folds": CV_FOLDS,
        "seeds": list(config.seeds),
    }
    deterministic, deterministic_metadata = run_deterministic(data, config.conditions)
    metadata.update(deterministic_metadata)
    scores = {
        **deterministic,
        **run_static_family(config, bank, split, metadata),
        **run_contextual(config, data, bank, split, metadata),
    }
    runs: list[RunResult] = []
    for name, entry in scores.items():
        add_runs(runs, name, entry, split)

    write_runs(config.output_dir / "runs.csv", runs)
    summary = summarize_runs(runs)
    write_summary(config.output_dir / "summary.csv", summary)
    write_contrasts(config.output_dir / "contrasts.json", scores, split)
    write_selected_predictions(config.output_dir / "val_row_predictions.csv", scores, summary, data)
    metadata["elapsed_seconds"] = round(perf_counter() - start, 2)
    metadata["packages"] = package_versions()
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(metadata, ensure_ascii=False, indent=2))


def summarize(
    condition: str, protocol: str, seed: int, matrix: np.ndarray, truth: np.ndarray
) -> list[RunResult]:
    """Turn one score matrix into a typed result row."""
    metrics = ranking_metrics(matrix, truth)
    return [
        RunResult(
            condition=condition,
            protocol=protocol,
            seed=seed,
            recall_at_1=metrics["recall@1"],
            recall_at_3=metrics["recall@3"],
            mrr=metrics["mrr"],
            n=int(metrics["n"]),
        )
    ]


@dataclass(frozen=True, slots=True)
class SummaryRow:
    """Aggregate of one condition on one protocol."""

    condition: str
    protocol: str
    runs: int
    recall_at_1_mean: float
    recall_at_1_sd: float
    recall_at_3_mean: float
    mrr_mean: float


def summarize_runs(rows: list[RunResult]) -> list[SummaryRow]:
    """Mean and seed spread per condition and protocol."""
    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.condition, row.protocol)].append(row)
    summary: list[SummaryRow] = []
    for (condition, protocol), records in sorted(grouped.items()):
        accuracy = np.array([record.recall_at_1 for record in records])
        summary.append(
            SummaryRow(
                condition=condition,
                protocol=protocol,
                runs=len(records),
                recall_at_1_mean=float(accuracy.mean()),
                recall_at_1_sd=float(accuracy.std(ddof=0)),
                recall_at_3_mean=float(np.mean([record.recall_at_3 for record in records])),
                mrr_mean=float(np.mean([record.mrr for record in records])),
            )
        )
    return summary


def write_runs(path: Path, rows: list[RunResult]) -> None:
    """Write every fitted run."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[SummaryRow]) -> None:
    """Write the per-condition aggregate."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_contrasts(path: Path, scores: dict[str, Scores], split: Split) -> None:
    """Write paired group-bootstrap intervals on both protocols."""
    records: list[dict[str, object]] = []
    for treatment, control in CONTRASTS:
        if treatment not in scores or control not in scores:
            continue
        for protocol in ("cv", "val"):
            left = scores[treatment].cv if protocol == "cv" else scores[treatment].val
            right = scores[control].cv if protocol == "cv" else scores[control].val
            shared = sorted(set(left) & set(right))
            if not shared:
                continue
            seed = min(shared)
            truth = split.train_truth if protocol == "cv" else split.val_truth
            mrr_interval = paired_bootstrap(
                per_row_reciprocal_rank(left[seed], truth),
                per_row_reciprocal_rank(right[seed], truth),
                truth,
            )
            top1_interval = paired_bootstrap(
                top1_hits(left[seed], truth), top1_hits(right[seed], truth), truth
            )
            records.append(
                {
                    "treatment": treatment,
                    "control": control,
                    "protocol": protocol,
                    "seed": seed,
                    "mrr_delta": mrr_interval["delta"],
                    "mrr_ci_low": mrr_interval["ci_low"],
                    "mrr_ci_high": mrr_interval["ci_high"],
                    "recall_at_1_delta": top1_interval["delta"],
                    "recall_at_1_ci_low": top1_interval["ci_low"],
                    "recall_at_1_ci_high": top1_interval["ci_high"],
                }
            )
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def top1_hits(matrix: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return per-row 0/1 top-1 correctness."""
    return (np.argmax(matrix, axis=1) == truth).astype(np.float64)


def write_selected_predictions(
    path: Path,
    scores: dict[str, Scores],
    summary: list[SummaryRow],
    data: LinkingData,
) -> None:
    """Export validation rankings for the condition with the best CV recall."""
    candidates = list(data.candidate_keys)
    cv_rows = [row for row in summary if row.protocol == "cv" and row.condition in scores]
    if not cv_rows:
        return
    best = max(cv_rows, key=lambda row: row.recall_at_1_mean)
    matrices = scores[best.condition].val
    if not matrices:
        return
    seed = min(matrices)
    matrix = matrices[seed]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "seed",
                "id",
                "true_indicator",
                "true_level",
                "top1",
                "top2",
                "top3",
                "top5",
            ]
        )
        for position, row in enumerate(data.val_rows):
            order = np.argsort(-matrix[position])[:5]
            writer.writerow(
                [
                    best.condition,
                    seed,
                    row.identifier,
                    row.indicator,
                    row.level if row.level is not None else "",
                    *(candidates[index] for index in order),
                ]
            )


def sha256(path: Path) -> str:
    """Stream a file hash without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Record the packages that decide the numbers."""
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "sentence-transformers",
        "fasttext-wheel",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    return packages


def main(
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_DIR,
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATA_DIR,
    cc_path: Annotated[Path, typer.Option()] = DEFAULT_CC,
    conditions: Annotated[str, typer.Option()] = ",".join(
        (*DETERMINISTIC, *ZEROSHOT_STATIC, *TRAINED_STATIC, "e5_proj")
    ),
    seeds: Annotated[str, typer.Option()] = ",".join(str(seed) for seed in SEEDS),
) -> None:
    """Run the requested linking conditions and write metrics.

    The environment variables NURI_EPOCHS and NURI_THREADS override the epoch
    count and the torch thread count, so the command line stays small.
    """
    requested = tuple(name.strip() for name in conditions.split(",") if name.strip())
    seed_tuple = tuple(int(value) for value in seeds.split(",") if value.strip())
    evaluate(
        RunConfig(
            output_dir=output_dir,
            data_dir=data_dir,
            cc_path=cc_path,
            conditions=requested,
            seeds=seed_tuple,
            epochs=int(os.environ.get("NURI_EPOCHS", str(EPOCHS))),
            learning_rate=LEARNING_RATE,
            threads=int(os.environ.get("NURI_THREADS", "4")),
        )
    )


if __name__ == "__main__":
    typer.run(main)
