"""PuriDivER-style replay purification for BrainUICL/ISRUC.

The original PuriDivER method operates on labeled image streams. BrainUICL's
target stream is unlabeled, so teacher pseudo labels take the role of observed
labels. Purification is performed per 30-second EEG epoch while replay remains
sequence based, matching BrainUICL's 20-epoch storage unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset


class IndexedBufferDataset(Dataset):
    """BrainUICL new/replay pairs with indices for purification metadata."""

    def __init__(self, new_paths, train_paths, train_len):
        self.new_data, self.new_label = new_paths
        self.train_data, self.train_label = train_paths
        self.train_len = train_len
        self.length = len(self.new_data)
        old_len = int(0.8 * self.length)
        new_len = self.length - old_len

        pseudo_count = len(self.train_data) - self.train_len
        if pseudo_count > 0 and new_len < pseudo_count:
            old_idx = np.random.choice(
                range(self.train_len), old_len, replace=self.train_len < old_len
            ).tolist()
            new_idx = np.random.choice(
                range(self.train_len, len(self.train_data)),
                new_len,
                replace=pseudo_count < new_len,
            ).tolist()
            self.sample_idx = old_idx + new_idx
        else:
            self.sample_idx = np.random.choice(
                range(len(self.train_data)),
                self.length,
                replace=len(self.train_data) < self.length,
            ).tolist()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_new = torch.from_numpy(np.load(self.new_data[index]).astype(np.float32))
        y_new = torch.from_numpy(np.load(self.new_label[index]).astype(np.int64))
        replay_idx = self.sample_idx[index]
        x_replay = torch.from_numpy(np.load(self.train_data[replay_idx]).astype(np.float32))
        y_replay = torch.from_numpy(np.load(self.train_label[replay_idx]).astype(np.int64))
        eog = torch.cat((x_new[:, :2, :], x_replay[:, :2, :]), dim=0)
        eeg = torch.cat((x_new[:, 2:, :], x_replay[:, 2:, :]), dim=0)
        label = torch.cat((y_new, y_replay), dim=0)
        return eog, eeg, label, index, replay_idx


class PathDataset(Dataset):
    def __init__(self, paths):
        self.data_paths, self.label_paths = paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.data_paths[index]).astype(np.float32))
        y = torch.from_numpy(np.load(self.label_paths[index]).astype(np.int64))
        return x[:, :2, :], x[:, 2:, :], y


def _forward(blocks, eog, eeg, args, return_features=False):
    batch = eeg.shape[0]
    eog_flat = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
    eeg_flat = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
    features = blocks[1](blocks[0](eeg_flat, eog_flat))
    logits = blocks[2](features).reshape(
        batch, args.model_param.NumClasses, args.model_param.SeqLength
    )
    if return_features:
        return logits, features
    return logits


def _set_train(blocks, train):
    for block in blocks:
        block.train(train)


def _low_component_probability(values, seed):
    """Posterior probability of the lower-mean component in a 2-GMM."""
    shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    result = np.ones_like(flat, dtype=np.float32)
    valid = flat[finite]
    if valid.size < 8 or float(valid.std()) < 1e-8:
        return result.reshape(shape)
    lo, hi = float(valid.min()), float(valid.max())
    normalized = (valid - lo) / max(hi - lo, 1e-12)
    try:
        gmm = GaussianMixture(
            n_components=2,
            max_iter=50,
            reg_covar=1e-5,
            random_state=seed,
        ).fit(normalized[:, None])
        low_component = int(np.argmin(gmm.means_.reshape(-1)))
        result[finite] = gmm.predict_proba(normalized[:, None])[:, low_component]
    except ValueError:
        result[finite] = 1.0
    return result.reshape(shape)


def _entropy(probabilities):
    probs = probabilities.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=1)


def _sharpen(probabilities, temperature):
    exponent = 1.0 / max(float(temperature), 1e-3)
    sharpened = probabilities.clamp_min(1e-8).pow(exponent)
    return sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def _collect_predictions(blocks, paths, args):
    loader = DataLoader(
        PathDataset(paths),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    probabilities, labels = [], []
    _set_train(blocks, False)
    for eog, eeg, label in loader:
        eog, eeg = eog.to(args.device), eeg.to(args.device)
        probabilities.append(_forward(blocks, eog, eeg, args).softmax(dim=1).cpu())
        labels.append(label.long())
    return torch.cat(probabilities, dim=0), torch.cat(labels, dim=0)


@dataclass
class PurificationState:
    new_clean: torch.Tensor
    new_low_uncertainty: torch.Tensor
    new_hard_targets: torch.Tensor
    new_soft_targets: torch.Tensor
    replay_clean: torch.Tensor
    replay_low_uncertainty: torch.Tensor
    replay_soft_targets: torch.Tensor
    summary: dict


def build_purification_state(new_paths, replay_paths, student_blocks, teacher_blocks, args):
    """Fit loss and uncertainty GMMs before adapting one target individual."""
    student_new, _ = _collect_predictions(student_blocks, new_paths, args)
    teacher_new, _ = _collect_predictions(teacher_blocks, new_paths, args)
    student_replay, replay_labels = _collect_predictions(student_blocks, replay_paths, args)

    new_hard = teacher_new.argmax(dim=1)
    new_nll = -student_new.gather(1, new_hard.unsqueeze(1)).squeeze(1).clamp_min(1e-8).log()
    new_clean = _low_component_probability(new_nll.numpy(), args.seed)
    teacher_entropy = _entropy(teacher_new) / np.log(args.model_param.NumClasses)
    mean_new = 0.5 * (teacher_new + student_new)
    disagreement = 0.5 * (
        (teacher_new * (teacher_new.clamp_min(1e-8).log() - mean_new.clamp_min(1e-8).log())).sum(dim=1)
        + (student_new * (student_new.clamp_min(1e-8).log() - mean_new.clamp_min(1e-8).log())).sum(dim=1)
    )
    new_uncertainty = teacher_entropy + disagreement
    new_low_unc = _low_component_probability(new_uncertainty.numpy(), args.seed + 1)

    replay_nll = -student_replay.gather(1, replay_labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8).log()
    replay_clean = _low_component_probability(replay_nll.numpy(), args.seed + 2)
    replay_uncertainty = _entropy(student_replay) / np.log(args.model_param.NumClasses)
    replay_low_unc = _low_component_probability(replay_uncertainty.numpy(), args.seed + 3)

    # Source replay labels are supervised ground truth. Only pseudo-labeled
    # target memory is potentially contaminated and should be purified.
    source_count = min(int(args.train_len), replay_clean.shape[0])
    replay_clean[:source_count] = 1.0
    replay_low_unc[:source_count] = 0.0

    new_soft = _sharpen(0.5 * (teacher_new + student_new), args.puridiver_soft_temperature)
    replay_soft = _sharpen(student_replay, args.puridiver_soft_temperature)

    new_clean_mask = new_clean >= args.puridiver_clean_threshold
    replay_clean_mask = replay_clean >= args.puridiver_clean_threshold
    new_relabel = (~new_clean_mask) & (new_low_unc >= args.puridiver_uncertainty_threshold)
    replay_relabel = (~replay_clean_mask) & (replay_low_unc >= args.puridiver_uncertainty_threshold)
    pseudo_slice = slice(source_count, None)
    pseudo_size = replay_clean[pseudo_slice].size
    summary = {
        "new_clean_rate": float(new_clean_mask.mean()),
        "new_relabel_rate": float(new_relabel.mean()),
        "new_unlabeled_rate": float((~new_clean_mask & ~new_relabel).mean()),
        "replay_clean_rate": float(replay_clean_mask.mean()),
        "replay_relabel_rate": float(replay_relabel.mean()),
        "replay_unlabeled_rate": float((~replay_clean_mask & ~replay_relabel).mean()),
        "pseudo_replay_clean_rate": (
            float(replay_clean_mask[pseudo_slice].mean()) if pseudo_size else 1.0
        ),
        "source_replay_protected": int(source_count),
        "new_mean_loss": float(new_nll.mean()),
        "replay_mean_loss": float(replay_nll.mean()),
    }
    return PurificationState(
        new_clean=torch.from_numpy(new_clean),
        new_low_uncertainty=torch.from_numpy(new_low_unc),
        new_hard_targets=new_hard,
        new_soft_targets=new_soft.permute(0, 2, 1).contiguous(),
        replay_clean=torch.from_numpy(replay_clean),
        replay_low_uncertainty=torch.from_numpy(replay_low_unc),
        replay_soft_targets=replay_soft.permute(0, 2, 1).contiguous(),
        summary=summary,
    )


def make_puridiver_loader(args, new_paths, shuffle=True):
    dataset = IndexedBufferDataset(new_paths, args.train_paths, args.train_len)
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_worker,
    )


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _classification_loss(hard_ce, soft_ce, clean_mask, relabel_mask, relabel_weight):
    clean_weight = clean_mask.to(hard_ce.dtype)
    relabel_weight_map = relabel_mask.to(hard_ce.dtype) * relabel_weight
    numerator = (hard_ce * clean_weight + soft_ce * relabel_weight_map).sum()
    denominator = (clean_weight + relabel_weight_map).sum().clamp_min(1.0)
    return numerator / denominator


def _augment(x, noise_ratio, scale_ratio):
    std = x.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
    scale_shape = (*x.shape[:-1], 1)
    scale = 1.0 + (2.0 * torch.rand(scale_shape, device=x.device) - 1.0) * scale_ratio
    return x * scale + torch.randn_like(x) * std * noise_ratio


class PuriDivEREEGFinetune:
    """Clean/relabel/consistency objective adapted to BrainUICL EEG streams."""

    def __init__(self, blocks, state, args):
        self.blocks = blocks
        self.state = state
        self.args = args
        self.optimizer = torch.optim.Adam(
            list(blocks[0].parameters()) + list(blocks[1].parameters()) + list(blocks[2].parameters()),
            lr=args.cl_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        self.last_diagnostics = {}

    def _metadata(self, new_indices, replay_indices, device):
        new_indices = new_indices.detach().cpu().long()
        replay_indices = replay_indices.detach().cpu().long()
        return {
            "new_clean": self.state.new_clean[new_indices].to(device),
            "new_low_unc": self.state.new_low_uncertainty[new_indices].to(device),
            "new_hard": self.state.new_hard_targets[new_indices].to(device),
            "new_soft": self.state.new_soft_targets[new_indices].to(device),
            "replay_clean": self.state.replay_clean[replay_indices].to(device),
            "replay_low_unc": self.state.replay_low_uncertainty[replay_indices].to(device),
            "replay_soft": self.state.replay_soft_targets[replay_indices].to(device),
        }

    def update(self, eeg, eog, labels, new_indices, replay_indices):
        seq_len = self.args.model_param.SeqLength
        eog_new, eog_replay = eog[:, :seq_len], eog[:, seq_len:]
        eeg_new, eeg_replay = eeg[:, :seq_len], eeg[:, seq_len:]
        replay_labels = labels[:, seq_len:].long()
        meta = self._metadata(new_indices, replay_indices, eeg.device)

        logits_new = _forward(self.blocks, eog_new, eeg_new, self.args)
        logits_replay, replay_features = _forward(
            self.blocks, eog_replay, eeg_replay, self.args, return_features=True
        )
        logits_new_t = logits_new.permute(0, 2, 1)
        logits_replay_t = logits_replay.permute(0, 2, 1)
        new_hard_ce = F.cross_entropy(
            logits_new_t.reshape(-1, self.args.model_param.NumClasses),
            meta["new_hard"].reshape(-1),
            reduction="none",
        ).reshape_as(meta["new_clean"])
        replay_hard_ce = F.cross_entropy(
            logits_replay_t.reshape(-1, self.args.model_param.NumClasses),
            replay_labels.reshape(-1),
            reduction="none",
        ).reshape_as(meta["replay_clean"])
        new_soft_ce = -(meta["new_soft"] * F.log_softmax(logits_new_t, dim=-1)).sum(dim=-1)
        replay_soft_ce = -(meta["replay_soft"] * F.log_softmax(logits_replay_t, dim=-1)).sum(dim=-1)

        clean_threshold = self.args.puridiver_clean_threshold
        uncertainty_threshold = self.args.puridiver_uncertainty_threshold
        new_clean = meta["new_clean"] >= clean_threshold
        replay_clean = meta["replay_clean"] >= clean_threshold
        new_relabel = (~new_clean) & (meta["new_low_unc"] >= uncertainty_threshold)
        replay_relabel = (~replay_clean) & (meta["replay_low_unc"] >= uncertainty_threshold)
        new_unlabeled = (~new_clean) & (~new_relabel)
        replay_unlabeled = (~replay_clean) & (~replay_relabel)

        eog_all = torch.cat((eog_new, eog_replay), dim=0)
        eeg_all = torch.cat((eeg_new, eeg_replay), dim=0)
        eog_strong = _augment(eog_all, self.args.puridiver_strong_noise, 0.08)
        eeg_strong = _augment(eeg_all, self.args.puridiver_strong_noise, 0.08)
        weak_prob = torch.cat((logits_new, logits_replay), dim=0).softmax(dim=1).detach()
        modes = [block.training for block in self.blocks]
        _set_train(self.blocks, False)
        strong_prob = _forward(self.blocks, eog_strong, eeg_strong, self.args).softmax(dim=1)
        for block, mode in zip(self.blocks, modes):
            block.train(mode)
        consistency = (strong_prob - weak_prob).pow(2).sum(dim=1)
        consistency_new, consistency_replay = consistency.chunk(2, dim=0)

        new_loss = (
            _classification_loss(
                new_hard_ce,
                new_soft_ce,
                new_clean,
                new_relabel,
                self.args.puridiver_relabel_weight,
            )
            + self.args.puridiver_consistency_weight * _masked_mean(consistency_new, new_unlabeled)
        )
        replay_loss = (
            _classification_loss(
                replay_hard_ce,
                replay_soft_ce,
                replay_clean,
                replay_relabel,
                self.args.puridiver_relabel_weight,
            )
            + self.args.puridiver_consistency_weight * _masked_mean(consistency_replay, replay_unlabeled)
        )
        loss = self.args.alpha * new_loss + (1.0 - self.args.alpha) * replay_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_diagnostics = {
            "new_loss": float(new_loss.detach().cpu()),
            "replay_loss": float(replay_loss.detach().cpu()),
            "consistency_loss": float(consistency.mean().detach().cpu()),
        }
        return float(loss.detach().cpu()), self.blocks, replay_features


@torch.no_grad()
def select_puridiver_memory(args, blocks):
    """Class-aware purity/diversity pruning with source data kept intact."""
    before = len(args.train_paths[0])
    budget = max(int(args.puridiver_memory_size), args.train_len)
    pseudo_count = before - args.train_len
    target_count = budget - args.train_len
    if pseudo_count <= target_count:
        return {"before": before, "after": before, "pruned": 0}

    pseudo_paths = (
        args.train_paths[0][args.train_len :],
        args.train_paths[1][args.train_len :],
    )
    loader = DataLoader(PathDataset(pseudo_paths), batch_size=args.batch, shuffle=False, num_workers=args.num_worker)
    losses, features, dominant_classes = [], [], []
    _set_train(blocks, False)
    for eog, eeg, labels in loader:
        eog, eeg, labels = eog.to(args.device), eeg.to(args.device), labels.to(args.device)
        logits, encoded = _forward(blocks, eog, eeg, args, return_features=True)
        ce = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, args.model_param.NumClasses),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape[0], -1).mean(dim=1)
        losses.append(ce.cpu())
        features.append(encoded.mean(dim=1).cpu())
        dominant_classes.append(torch.mode(labels, dim=1).values.cpu())
    losses = torch.cat(losses).numpy()
    features = F.normalize(torch.cat(features), dim=1)
    classes = torch.cat(dominant_classes).numpy()
    clean_prob = _low_component_probability(losses, args.seed + before).reshape(-1)

    selected = []
    class_quota = max(1, target_count // args.model_param.NumClasses)
    for class_id in range(args.model_param.NumClasses):
        candidates = np.flatnonzero(classes == class_id).tolist()
        quota = min(class_quota, len(candidates))
        class_selected = []
        while candidates and len(class_selected) < quota:
            if not class_selected:
                chosen = max(candidates, key=lambda idx: clean_prob[idx])
            else:
                similarity = features[candidates] @ features[class_selected].T
                redundancy = similarity.max(dim=1).values.add(1.0).mul(0.5).numpy()
                purity_cost = 1.0 - clean_prob[candidates]
                score = (
                    args.puridiver_purity_weight * purity_cost
                    + (1.0 - args.puridiver_purity_weight) * redundancy
                )
                chosen = candidates[int(np.argmin(score))]
            candidates.remove(chosen)
            class_selected.append(chosen)
        selected.extend(class_selected)

    if len(selected) < target_count:
        remaining = [idx for idx in range(pseudo_count) if idx not in set(selected)]
        remaining.sort(key=lambda idx: clean_prob[idx], reverse=True)
        selected.extend(remaining[: target_count - len(selected)])
    selected = sorted(selected[:target_count])
    source_data = args.train_paths[0][: args.train_len]
    source_labels = args.train_paths[1][: args.train_len]
    args.train_paths[0] = source_data + [pseudo_paths[0][idx] for idx in selected]
    args.train_paths[1] = source_labels + [pseudo_paths[1][idx] for idx in selected]
    return {
        "before": before,
        "after": len(args.train_paths[0]),
        "pruned": before - len(args.train_paths[0]),
        "candidate_clean_probability": float(clean_prob.mean()),
        "selected_clean_probability": float(clean_prob[selected].mean()) if selected else 0.0,
    }
