"""EEG adaptation of Self-Purified Replay's Self-Centered filter.

The filter never consumes ground-truth labels. It groups expert embeddings by
the model's pseudo-label, estimates stochastic eigenvector centrality, and fits
a two-component beta mixture to obtain a clean posterior for each EEG epoch.
Sequence-level acceptance aggregates those epoch posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


@dataclass(frozen=True)
class PurificationResult:
    accepted: np.ndarray
    candidate: np.ndarray
    sequence_scores: np.ndarray
    epoch_clean_probabilities: np.ndarray
    clean_epoch_counts: np.ndarray


class BetaMixture1D:
    """Numerically guarded two-component beta-mixture EM."""

    def __init__(self, max_iters: int = 10):
        self.max_iters = max_iters
        self.alphas = np.array([1.0, 2.0], dtype=np.float64)
        self.betas = np.array([2.0, 1.0], dtype=np.float64)
        self.weights = np.array([0.5, 0.5], dtype=np.float64)
        self.eps = 1e-8

    def _likelihood(self, x: np.ndarray, component: int) -> np.ndarray:
        values = stats.beta.pdf(x, self.alphas[component], self.betas[component])
        return np.nan_to_num(values, nan=self.eps, posinf=1.0 / self.eps, neginf=self.eps)

    def _responsibilities(self, x: np.ndarray) -> np.ndarray:
        weighted = np.stack([self.weights[k] * self._likelihood(x, k) for k in range(2)])
        weighted = np.maximum(weighted, self.eps)
        return weighted / weighted.sum(axis=0, keepdims=True)

    def fit(self, x: np.ndarray) -> "BetaMixture1D":
        x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        for _ in range(self.max_iters):
            responsibilities = self._responsibilities(x)
            for component in range(2):
                weights = responsibilities[component]
                weight_sum = max(float(weights.sum()), self.eps)
                mean = float(np.sum(weights * x) / weight_sum)
                variance = float(np.sum(weights * (x - mean) ** 2) / weight_sum)
                variance = max(variance, 1e-5)
                concentration = mean * (1.0 - mean) / variance - 1.0
                if not np.isfinite(concentration) or concentration <= 0:
                    concentration = 2.0
                self.alphas[component] = max(mean * concentration, 0.1)
                self.betas[component] = max((1.0 - mean) * concentration, 0.1)
            self.weights = responsibilities.sum(axis=1)
            self.weights /= self.weights.sum()
        return self

    def clean_posterior(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        component_means = self.alphas / (self.alphas + self.betas)
        clean_component = int(np.argmax(component_means))
        return self._responsibilities(x)[clean_component]


def _normalized_centrality(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [5, 95])
    normalized = (values - low) / (high - low + 1e-8)
    return np.clip(normalized, 1e-4, 1.0 - 1e-4)


def _power_centrality(adjacency: np.ndarray, max_iters: int = 200) -> np.ndarray:
    count = adjacency.shape[0]
    vector = np.full(count, 1.0 / np.sqrt(count), dtype=np.float64)
    for _ in range(max_iters):
        updated = adjacency @ vector
        norm = np.linalg.norm(updated)
        if norm <= 1e-12:
            return np.ones(count, dtype=np.float64)
        updated /= norm
        if np.linalg.norm(updated - vector) <= 1e-7:
            vector = updated
            break
        vector = updated
    return np.abs(vector)


def _class_clean_probabilities(
    features: np.ndarray,
    ensembles: int,
    bmm_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = features.shape[0]
    if count < 8:
        return np.ones(count, dtype=np.float64)

    normalized_features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    similarity = np.clip(normalized_features @ normalized_features.T, 0.0, 1.0)
    np.fill_diagonal(similarity, 0.0)
    ensemble_posteriors = []
    for _ in range(ensembles):
        random_upper = rng.random((count, count))
        random_upper = np.triu(random_upper, 1)
        random_symmetric = random_upper + random_upper.T
        adjacency = (similarity > random_symmetric).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)
        centrality = _normalized_centrality(_power_centrality(adjacency))
        if float(np.std(centrality)) < 1e-5:
            ensemble_posteriors.append(np.ones(count, dtype=np.float64))
            continue
        try:
            posterior = BetaMixture1D(max_iters=bmm_iters).fit(centrality).clean_posterior(centrality)
            if not np.all(np.isfinite(posterior)):
                raise FloatingPointError("non-finite beta-mixture posterior")
        except (FloatingPointError, ValueError):
            posterior = centrality
        ensemble_posteriors.append(posterior)
    return np.mean(np.stack(ensemble_posteriors), axis=0)


def self_centered_clean_probabilities(
    features: np.ndarray,
    observed_labels: np.ndarray,
    *,
    ensembles: int = 5,
    bmm_iters: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Estimate SPR clean posteriors without a classifier-confidence gate."""

    features = np.asarray(features, dtype=np.float32)
    observed_labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != observed_labels.shape[0]:
        raise ValueError("expected features [N,D] and observed_labels [N]")
    clean = np.zeros(features.shape[0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    for class_id in np.unique(observed_labels):
        indices = np.flatnonzero(observed_labels == class_id)
        clean[indices] = _class_clean_probabilities(
            features[indices], ensembles=ensembles, bmm_iters=bmm_iters, rng=rng
        )
    return clean


def symmetric_label_noise(
    labels: np.ndarray,
    rate: float,
    num_classes: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace selected labels uniformly with a guaranteed different class."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("noise rate must be in [0, 1]")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    labels = np.asarray(labels, dtype=np.int64)
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("labels contain an invalid class")
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    mask = rng.random(labels.shape) < rate
    offsets = rng.integers(1, num_classes, size=labels.shape)
    noisy[mask] = (labels[mask] + offsets[mask]) % num_classes
    return noisy, mask


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss for two aligned batches of projected embeddings."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 2 or first.shape != second.shape:
            raise ValueError("expected two embedding tensors with shape [B,D]")
        if first.shape[0] < 2:
            raise ValueError("NT-Xent requires at least two paired samples")
        batch = first.shape[0]
        embeddings = F.normalize(torch.cat((first, second), dim=0), dim=1)
        logits = embeddings @ embeddings.T / self.temperature
        diagonal = torch.eye(2 * batch, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        targets = (torch.arange(2 * batch, device=logits.device) + batch) % (2 * batch)
        return F.cross_entropy(logits, targets)


@dataclass(frozen=True)
class EpochMemoryRecord:
    data_path: str
    epoch_index: int
    observed_label: int
    clean_probability: float
    true_label: int = -1


class PurifiedEpochBuffer:
    """Fixed EEG memory with a protected source partition and SPR dynamic partition."""

    def __init__(self, capacity: int, source_capacity: int, num_classes: int):
        if capacity <= 0 or not 0 <= source_capacity <= capacity:
            raise ValueError("invalid memory capacity")
        self.capacity = int(capacity)
        self.source_capacity = int(source_capacity)
        self.dynamic_capacity = self.capacity - self.source_capacity
        self.num_classes = int(num_classes)
        self.source: list[EpochMemoryRecord] = []
        self.dynamic: list[EpochMemoryRecord] = []
        self.stream_counts = np.zeros(self.num_classes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.source) + len(self.dynamic)

    @property
    def records(self) -> list[EpochMemoryRecord]:
        return self.source + self.dynamic

    def seed_source(self, records: list[EpochMemoryRecord], *, seed: int) -> None:
        """Seed a sequence-aware fixed source partition once."""

        if self.source:
            raise RuntimeError("source memory has already been seeded")
        rng = np.random.default_rng(seed)
        grouped: dict[str, list[EpochMemoryRecord]] = {}
        for record in records:
            grouped.setdefault(record.data_path, []).append(record)
        paths = list(grouped)
        rng.shuffle(paths)
        selected: list[EpochMemoryRecord] = []
        for path in paths:
            candidates = grouped[path]
            remaining = self.source_capacity - len(selected)
            if remaining <= 0:
                break
            if len(candidates) <= remaining:
                selected.extend(candidates)
            else:
                indices = rng.choice(len(candidates), size=remaining, replace=False)
                selected.extend(candidates[int(index)] for index in indices)
        self.source = selected[: self.source_capacity]

    def update(self, records: list[EpochMemoryRecord]) -> None:
        """Apply class-aware purified reservoir replacement to accepted epochs."""

        if self.dynamic_capacity == 0:
            return
        for record in records:
            label = int(record.observed_label)
            if not 0 <= label < self.num_classes:
                raise ValueError("record contains an invalid class")
            self.stream_counts[label] += 1
            self.dynamic.append(record)
            if len(self.dynamic) <= self.dynamic_capacity:
                continue
            class_sizes = np.bincount(
                [item.observed_label for item in self.dynamic], minlength=self.num_classes
            ).astype(np.float64)
            total_seen = max(int(self.stream_counts.sum()), 1)
            target = self.dynamic_capacity * self.stream_counts / total_seen
            evict_class = int(np.argmax(class_sizes - target))
            candidates = [
                index for index, item in enumerate(self.dynamic)
                if item.observed_label == evict_class
            ]
            evict_index = min(candidates, key=lambda index: self.dynamic[index].clean_probability)
            self.dynamic.pop(evict_index)

    def class_counts(self) -> list[int]:
        return np.bincount(
            [record.observed_label for record in self.records], minlength=self.num_classes
        ).astype(int).tolist()


def purify_eeg_sequences(
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    confidences: np.ndarray,
    *,
    confidence_threshold: float,
    min_confident_epochs: int,
    clean_threshold: float,
    min_clean_epochs: int,
    sequence_threshold: float,
    ensembles: int,
    bmm_iters: int,
    seed: int,
) -> PurificationResult:
    """Select sequences using only expert features and model predictions."""

    features = np.asarray(features, dtype=np.float32)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if features.ndim != 3 or pseudo_labels.shape != confidences.shape or features.shape[:2] != pseudo_labels.shape:
        raise ValueError("expected features [N,S,D] and labels/confidences [N,S]")

    confident = confidences >= confidence_threshold
    candidate = confident.sum(axis=1) >= min_confident_epochs
    epoch_clean = np.zeros_like(confidences, dtype=np.float64)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_labels = pseudo_labels.reshape(-1)
    graph_mask = (confident & candidate[:, None]).reshape(-1)
    rng = np.random.default_rng(seed)

    for class_id in np.unique(flat_labels[graph_mask]):
        indices = np.flatnonzero(graph_mask & (flat_labels == class_id))
        epoch_clean.reshape(-1)[indices] = _class_clean_probabilities(
            flat_features[indices], ensembles=ensembles, bmm_iters=bmm_iters, rng=rng
        )

    clean_epoch_counts = ((epoch_clean >= clean_threshold) & confident).sum(axis=1)
    sequence_scores = np.zeros(features.shape[0], dtype=np.float64)
    for index in np.flatnonzero(candidate):
        sequence_scores[index] = float(epoch_clean[index, confident[index]].mean())
    accepted = candidate & (clean_epoch_counts >= min_clean_epochs) & (sequence_scores >= sequence_threshold)
    return PurificationResult(
        accepted=accepted,
        candidate=candidate,
        sequence_scores=sequence_scores,
        epoch_clean_probabilities=epoch_clean,
        clean_epoch_counts=clean_epoch_counts,
    )
