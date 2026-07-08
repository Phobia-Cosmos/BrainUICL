#!/usr/bin/env python3
"""
Independent BrainUICL continual-learning runner with RTTDP-style attacks.

This script intentionally does not modify or call the original trainer/trainer.py
entrypoint. It writes all checkpoints, pseudo labels, logs and metrics under an
experiment output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.pretrain_net import FeatureExtractor, SleepMLP, TransformerEncoder  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from utils.util import compute_aaf1, compute_aaa, compute_forget, fix_randomness  # noqa: E402
from utils.util_block import MultiHeadAttentionBlock  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, paths: tuple[list[Path], list[Path]]):
        self.data_paths, self.label_paths = paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x = np.load(self.data_paths[index]).astype(np.float32)
        y = np.load(self.label_paths[index]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x[:, :2, :], x[:, 2:, :], y


class BufferDataset(Dataset):
    def __init__(self, new_paths, train_paths, train_len):
        self.new_data, self.new_label = new_paths
        self.train_data, self.train_label = train_paths
        self.train_len = train_len
        self.len = len(self.new_data)
        self.old_len = int(0.8 * self.len)
        self.new_len = int(0.2 * self.len)

        pseudo_count = len(self.train_data) - self.train_len
        if pseudo_count > 0 and self.new_len < pseudo_count:
            old_idx = list(np.random.choice(range(self.train_len), self.old_len, replace=self.train_len < self.old_len))
            new_idx = list(
                np.random.choice(
                    range(self.train_len, len(self.train_data)),
                    self.new_len,
                    replace=pseudo_count < self.new_len,
                )
            )
            sample_idx = old_idx + new_idx
        else:
            sample_idx = list(np.random.choice(range(len(self.train_data)), self.len, replace=len(self.train_data) < self.len))
        while len(sample_idx) < self.len:
            sample_idx.append(int(np.random.choice(range(self.train_len), 1, replace=False)[0]))
        self.sample_idx = sample_idx[: self.len]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_new = torch.from_numpy(np.load(self.new_data[index]).astype(np.float32))
        y_new = torch.from_numpy(np.load(self.new_label[index]).astype(np.int64))
        train_idx = self.sample_idx[index]
        x_train = torch.from_numpy(np.load(self.train_data[train_idx]).astype(np.float32))
        y_train = torch.from_numpy(np.load(self.train_label[train_idx]).astype(np.int64))

        eog = torch.cat((x_new[:, :2, :], x_train[:, :2, :]), dim=0)
        eeg = torch.cat((x_new[:, 2:, :], x_train[:, 2:, :]), dim=0)
        label = torch.cat((y_new, y_train), dim=0)
        return eog, eeg, label


def subject_paths(data_root: Path, subject: int) -> tuple[list[Path], list[Path]]:
    data_dir = data_root / str(subject) / "data"
    label_dir = data_root / str(subject) / "label"
    data_paths, label_paths = [], []
    idx = 0
    while (data_dir / f"{idx}.npy").exists():
        data_paths.append(data_dir / f"{idx}.npy")
        label_paths.append(label_dir / f"{idx}.npy")
        idx += 1
    return data_paths, label_paths


def merge_subject_paths(data_root: Path, subjects: list[int]) -> tuple[list[Path], list[Path]]:
    data_paths, label_paths = [], []
    for subject in subjects:
        d, l = subject_paths(data_root, subject)
        data_paths.extend(d)
        label_paths.extend(l)
    return data_paths, label_paths


def discover_subjects(data_root: Path) -> list[int]:
    subjects = []
    for sid in range(1, 101):
        if sid in (8, 40):
            continue
        if (data_root / str(sid) / "data" / "0.npy").exists():
            subjects.append(sid)
    if len(subjects) < 5:
        raise RuntimeError(f"Need at least 5 processed subjects under {data_root}, found {len(subjects)}")
    return subjects


def split_subjects(subjects: list[int], seed: int):
    fix_randomness(seed)
    idx = list(subjects)
    path_len = len(idx)
    old_count = max(1, int(path_len * 0.2))
    new_count = max(1, int(path_len * 0.5))
    new_count = min(new_count, path_len - old_count - 2)
    old_task_idx = list(np.random.choice(idx, old_count, replace=False))
    remaining_idx = sorted(set(idx) - set(old_task_idx))
    new_task_idx = list(np.random.choice(remaining_idx, new_count, replace=False))
    train_val_idx = sorted(set(idx) - set(old_task_idx) - set(new_task_idx))
    train_count = max(1, int(len(train_val_idx) * 0.8))
    train_count = min(train_count, len(train_val_idx) - 1)
    train_idx = list(np.random.choice(train_val_idx, train_count, replace=False))
    val_idx = [i for i in train_val_idx if i not in train_idx]
    return train_idx, val_idx, old_task_idx, new_task_idx


def build_blocks(args):
    return (
        FeatureExtractor(args).float().to(args.device),
        TransformerEncoder(args).float().to(args.device),
        SleepMLP(args).float().to(args.device),
    )


def load_pretrained(args):
    blocks = build_blocks(args)
    ckpt_dir = args.input_checkpoint_root / args.dataset / "Pretrain"
    blocks[0].load_state_dict(torch.load(ckpt_dir / f"feature_extractor_parameter_{args.seed}.pkl", map_location=args.device))
    blocks[1].load_state_dict(torch.load(ckpt_dir / f"feature_encoder_parameter_{args.seed}.pkl", map_location=args.device))
    blocks[2].load_state_dict(torch.load(ckpt_dir / f"sleep_classifier_parameter_{args.seed}.pkl", map_location=args.device))
    return blocks


def save_blocks(blocks, save_dir: Path, seed: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    names = ["feature_extractor", "feature_encoder", "sleep_classifier"]
    for name, block in zip(names, blocks):
        state = {key: value.detach().cpu() for key, value in block.state_dict().items()}
        torch.save(state, save_dir / f"{name}_parameter_{seed}.pkl")


def clone_blocks(blocks, args):
    return tuple(copy.deepcopy(block).to(args.device) for block in blocks)


def set_train(blocks, train: bool):
    for block in blocks:
        block.train(train)


def forward_blocks(blocks, eog, eeg, args):
    batch = eeg.shape[0]
    eog = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
    eeg = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
    features = blocks[0](eeg, eog)
    features = blocks[1](features)
    logits = blocks[2](features)
    return logits.reshape(batch, args.model_param.NumClasses, args.model_param.SeqLength)


def feature_embeddings(blocks, eog, eeg, args):
    batch = eeg.shape[0]
    eog = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
    eeg = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
    features = blocks[0](eeg, eog)
    features = blocks[1](features)
    return features.reshape(batch, args.model_param.SeqLength, -1).mean(dim=1)


def flat_logits(logits):
    return logits.permute(0, 2, 1).reshape(-1, logits.shape[1])


def flat_labels(labels):
    return labels.reshape(-1).long()


@torch.no_grad()
def evaluate(blocks, loader, args, max_batches=0):
    set_train(blocks, False)
    y_true, y_pred = [], []
    predictions = []
    for batch_idx, (eog, eeg, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        eog, eeg, labels = eog.to(args.device), eeg.to(args.device), labels.to(args.device)
        logits = forward_blocks(blocks, eog, eeg, args)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        y_true.extend(flat_labels(labels).detach().cpu().tolist())
        y_pred.extend(pred.reshape(-1).detach().cpu().tolist())
        predictions.append(logits.detach().cpu())
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "mf1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)),
        "n_epochs": len(y_true),
        "predictions": torch.cat(predictions, dim=0) if predictions else None,
    }


def make_loader(data_root, subjects, batch, shuffle, num_workers=0):
    return DataLoader(
        SequenceDataset(merge_subject_paths(data_root, subjects)),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def sample_paths(paths, max_items):
    data_paths, label_paths = paths
    if max_items <= 0 or len(data_paths) <= max_items:
        return list(data_paths), list(label_paths)
    idx = np.random.choice(range(len(data_paths)), max_items, replace=False)
    return [data_paths[i] for i in idx], [label_paths[i] for i in idx]


@torch.no_grad()
def estimate_feature_centroid(blocks, paths, args, max_batches):
    loader = DataLoader(SequenceDataset(paths), batch_size=args.batch, shuffle=True, num_workers=0)
    vectors = []
    set_train(blocks, False)
    for batch_idx, (eog, eeg, _labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        eog, eeg = eog.to(args.device), eeg.to(args.device)
        vectors.append(feature_embeddings(blocks, eog, eeg, args).detach())
    if not vectors:
        return None
    return torch.cat(vectors, dim=0).mean(dim=0)


def update_source_drift_direction(args, teacher_blocks, new_task_id, attack_state):
    replay_paths = sample_paths(args.train_paths, args.stealth_centroid_batches * args.batch)
    new_paths = sample_paths(subject_paths(args.data_root, new_task_id), args.stealth_centroid_batches * args.batch)
    replay_centroid = estimate_feature_centroid(teacher_blocks, replay_paths, args, args.stealth_centroid_batches)
    new_centroid = estimate_feature_centroid(teacher_blocks, new_paths, args, args.stealth_centroid_batches)
    if replay_centroid is None or new_centroid is None:
        return
    attack_state.set_drift_direction(new_centroid - replay_centroid, args.stealth_direction_momentum)


def update_loss_drift_direction(args, teacher_blocks, attack_state):
    replay_paths = sample_paths(args.train_paths, args.stealth_centroid_batches * args.batch)
    loader = DataLoader(SequenceDataset(replay_paths), batch_size=args.batch, shuffle=True, num_workers=0)
    set_train(teacher_blocks, False)
    direction_sum = None
    count = 0
    for batch_idx, (eog, eeg, labels) in enumerate(loader):
        if args.stealth_centroid_batches and batch_idx >= args.stealth_centroid_batches:
            break
        eog, eeg, labels = eog.to(args.device), eeg.to(args.device), labels.to(args.device)
        batch = eeg.shape[0]
        eog_f = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
        eeg_f = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
        features = teacher_blocks[0](eeg_f, eog_f)
        features = teacher_blocks[1](features)
        features.retain_grad()
        logits = teacher_blocks[2](features).reshape(batch, args.model_param.NumClasses, args.model_param.SeqLength)
        loss = F.cross_entropy(flat_logits(logits), flat_labels(labels))
        grad = torch.autograd.grad(loss, features, retain_graph=False, create_graph=False)[0]
        direction = grad.reshape(-1, grad.shape[-1]).mean(dim=0).detach()
        direction_sum = direction if direction_sum is None else direction_sum + direction
        count += 1
    if direction_sum is not None and count > 0:
        attack_state.set_drift_direction(direction_sum / count, args.stealth_direction_momentum)


def get_uploaded_subject_paths(args, subject, use_uploaded=True):
    if use_uploaded and hasattr(args, "uploaded_subject_paths") and int(subject) in args.uploaded_subject_paths:
        return args.uploaded_subject_paths[int(subject)]
    return subject_paths(args.data_root, subject)


def make_new_loader(args, subject, is_buffer, shuffle, use_uploaded=True):
    new_paths = get_uploaded_subject_paths(args, subject, use_uploaded=use_uploaded)
    if is_buffer:
        dataset = BufferDataset(new_paths, args.train_paths, args.train_len)
    else:
        dataset = SequenceDataset(new_paths)
    return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle, num_workers=args.num_worker)


class CPCProbe:
    def __init__(self, blocks, args):
        self.args = args
        self.feature_extractor, self.feature_encoder, self.classifier = blocks
        self.d_model = args.model_param.EncoderParam.d_model
        self.timestep = 3
        self.Wk = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4),
                    nn.Dropout(0.1),
                    nn.GELU(),
                    nn.Linear(self.d_model * 4, self.d_model),
                ).to(args.device)
                for _ in range(self.timestep)
            ]
        )
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = MultiHeadAttentionBlock(
            self.d_model,
            args.model_param.EncoderParam.layer_num,
            args.model_param.EncoderParam.drop,
            args.model_param.EncoderParam.n_head,
        ).to(args.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.feature_extractor.parameters())},
                {"params": list(self.feature_encoder.parameters())},
                {"params": list(self.encoder.parameters()), "lr": args.lr},
                {"params": list(self.Wk.parameters()), "lr": args.lr},
            ],
            lr=args.ssl_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

    def update(self, eeg, eog):
        seq_len = self.args.model_param.SeqLength
        batch = eeg.shape[0]
        eog = eog.reshape(-1, self.args.model_param.EogNum, self.args.model_param.EpochLength)
        eeg = eeg.reshape(-1, self.args.model_param.EegNum, self.args.model_param.EpochLength)
        features = self.feature_extractor(eeg, eog)
        features = self.feature_encoder(features)
        t_samples = torch.randint(low=10, high=(seq_len - self.timestep), size=(1,), device=self.args.device).long()
        encode_samples = torch.empty((self.timestep, batch, self.d_model), device=self.args.device)
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = features[:, t_samples + i, :].view(batch, self.d_model)
        forward_seq = features[:, : t_samples + 1, :]
        output = self.encoder(forward_seq)
        c_t = output[:, t_samples, :].view(batch, -1)
        pred = torch.empty((self.timestep, batch, self.d_model), device=self.args.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)
        loss = 0
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1.0 * batch * self.timestep
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier)


class BufferPseudoLabelFinetuneProbe:
    def __init__(self, blocks, teacher_blocks, args):
        self.args = args
        self.feature_extractor, self.feature_encoder, self.classifier = [block.to(args.device) for block in blocks]
        self.feature_extractor_t, self.feature_encoder_t, self.classifier_t = [block.to(args.device) for block in teacher_blocks]
        self.softmax = nn.Softmax(dim=1)
        self.confidence_level = args.confidence
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.feature_extractor.parameters())},
                {"params": list(self.feature_encoder.parameters())},
                {"params": list(self.classifier.parameters())},
            ],
            lr=args.cl_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, eeg, eog, label):
        batch = eeg.shape[0]
        seq_len = self.args.model_param.SeqLength
        epoch_size = self.args.model_param.EpochLength
        eog_new, eog_train = eog[:, :seq_len, :, :], eog[:, seq_len:, :, :]
        eeg_new, eeg_train = eeg[:, :seq_len, :, :], eeg[:, seq_len:, :, :]
        eog_all = torch.cat((eog_new, eog_train), dim=0).reshape(-1, self.args.model_param.EogNum, epoch_size)
        eeg_all = torch.cat((eeg_new, eeg_train), dim=0).reshape(-1, self.args.model_param.EegNum, epoch_size)
        label_train = label[:, seq_len:]
        features = self.feature_extractor(eeg_all, eog_all)
        features = self.feature_encoder(features)
        features_train = features[batch:, :, :]
        features_new = features[:batch, :, :]
        pred_train = self.classifier(features_train)

        with torch.no_grad():
            eog_teacher = eog_new.contiguous().reshape(-1, self.args.model_param.EogNum, epoch_size)
            eeg_teacher = eeg_new.contiguous().reshape(-1, self.args.model_param.EegNum, epoch_size)
            teacher_features = self.feature_extractor_t(eeg_teacher, eog_teacher)
            teacher_features = self.feature_encoder_t(teacher_features)
            teacher_pred = self.classifier_t(teacher_features)
            teacher_pred = teacher_pred.permute(0, 2, 1).reshape(-1, self.args.model_param.NumClasses)
            teacher_prob = self.softmax(teacher_pred)
            pred_prob = teacher_prob.max(1, keepdim=True)[0].squeeze()
            target_pseudo = teacher_prob.max(1, keepdim=True)[1].squeeze()

        pred_target = self.classifier(features_new).permute(0, 2, 1).reshape(-1, self.args.model_param.NumClasses)
        confident_pred = pred_target[pred_prob > self.confidence_level]
        confident_labels = target_pseudo[pred_prob > self.confidence_level]
        clean_confident_labels = confident_labels.long()
        if confident_pred.shape[0] == 0:
            loss_new = pred_target.sum() * 0.0
            loss_new_ascent = pred_target.sum() * 0.0
        else:
            if self.args.attack_mode.startswith("stealth_") and self.args.stealth_train_new_bias_rate > 0:
                confident_scores = teacher_prob[pred_prob > self.confidence_level]
                confident_labels = bias_labels(
                    confident_labels,
                    confident_scores,
                    self.args.stealth_train_new_bias_rate,
                    self.args.stealth_train_bias_mode,
                    self.args.model_param.NumClasses,
                )
            loss_new = self.cross_entropy(confident_pred, confident_labels.long())
            loss_new_ascent = self.cross_entropy(confident_pred, clean_confident_labels)
        train_labels = label_train.long()
        loss_old_clean = self.cross_entropy(pred_train, train_labels)
        if self.args.attack_mode.startswith("stealth_") and self.args.stealth_train_replay_bias_rate > 0:
            train_labels = bias_labels(
                train_labels,
                pred_train,
                self.args.stealth_train_replay_bias_rate,
                self.args.stealth_train_bias_mode,
                self.args.model_param.NumClasses,
            )
        loss_old = self.cross_entropy(pred_train, train_labels)
        new_loss_weight = self.args.alpha
        if self.args.attack_mode.startswith("stealth_"):
            new_loss_weight *= self.args.stealth_train_new_loss_scale
        loss = new_loss_weight * loss_new + (1 - self.args.alpha) * loss_old
        if self.args.attack_mode.startswith("stealth_"):
            loss = loss - self.args.stealth_new_ascent_weight * loss_new_ascent
            loss = loss - self.args.stealth_replay_ascent_weight * loss_old_clean
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier), features_train


class AttackState:
    def __init__(self, class_num, device):
        self.class_num = class_num
        self.device = device
        self.class_prob = torch.ones(class_num, class_num, device=device) / class_num
        self.momentum = 0.8
        self.drift_direction = None
        self.stats = self.empty_stats()

    def empty_stats(self):
        return {
            "attempted": 0,
            "accepted": 0,
            "clean_pass": 0,
            "adv_pass": 0,
            "mean_rel_eog": [],
            "mean_rel_eeg": [],
            "mean_feature_shift": [],
            "mean_proxy_conflict": [],
            "mean_proxy_update_norm": [],
            "mean_proxy_old_norm": [],
            "mean_proxy_confidence": [],
        }

    def nhe_target(self, logits):
        pred = logits.detach().argmax(dim=1)
        target = torch.ones_like(logits)
        target[torch.arange(pred.shape[0], device=logits.device), pred] = 0.0
        return target / target.sum(dim=1, keepdim=True)

    def ble_target(self, logits):
        probs = logits.detach().softmax(dim=1)
        pred = probs.argmax(dim=1)
        new_prob = self.class_prob.clone()
        for cls in pred.unique():
            mask = pred == cls
            new_prob[cls] = self.momentum * new_prob[cls] + (1 - self.momentum) * probs[mask].mean(dim=0)
        score = new_prob.clone()
        score[torch.eye(self.class_num, device=logits.device).bool()] = 0.0
        mapping = score.argmax(dim=1)
        target = torch.zeros_like(logits)
        target[torch.arange(pred.shape[0], device=logits.device), mapping[pred]] = 1.0
        self.class_prob = new_prob.detach()
        return target

    def get_drift_direction(self, feature_dim):
        if self.drift_direction is None or self.drift_direction.numel() != feature_dim:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(20260707)
            direction = torch.randn(feature_dim, generator=generator, device=self.device)
            self.drift_direction = direction / direction.norm().clamp_min(1e-12)
        return self.drift_direction

    def set_drift_direction(self, direction, momentum):
        direction = direction.detach().to(self.device)
        direction = direction / direction.norm().clamp_min(1e-12)
        if self.drift_direction is None or self.drift_direction.numel() != direction.numel():
            self.drift_direction = direction
        else:
            mixed = momentum * self.drift_direction + (1 - momentum) * direction
            self.drift_direction = mixed / mixed.norm().clamp_min(1e-12)

    def record_stealth(self, clean_pass, adv_pass, accepted, rel_eog, rel_eeg, feature_shift):
        self.stats["attempted"] += int(clean_pass.numel())
        self.stats["clean_pass"] += int(clean_pass.sum().detach().cpu())
        self.stats["adv_pass"] += int(adv_pass.sum().detach().cpu())
        self.stats["accepted"] += int(accepted.sum().detach().cpu())
        self.stats["mean_rel_eog"].append(float(rel_eog.detach().mean().cpu()))
        self.stats["mean_rel_eeg"].append(float(rel_eeg.detach().mean().cpu()))
        self.stats["mean_feature_shift"].append(float(feature_shift.detach().mean().cpu()))

    def record_proxy_meta(self, clean_pass, adv_pass, rel_eog, rel_eeg, feature_shift, conflict, update_norm, old_norm, confidence):
        self.record_stealth(clean_pass, adv_pass, adv_pass, rel_eog, rel_eeg, feature_shift)
        self.stats["mean_proxy_conflict"].append(float(conflict.detach().cpu()))
        self.stats["mean_proxy_update_norm"].append(float(update_norm.detach().cpu()))
        self.stats["mean_proxy_old_norm"].append(float(old_norm.detach().cpu()))
        self.stats["mean_proxy_confidence"].append(float(confidence.detach().mean().cpu()))

    def flush_stats(self):
        attempted = max(self.stats["attempted"], 1)
        payload = {
            "attempted": int(self.stats["attempted"]),
            "clean_pass_rate": float(self.stats["clean_pass"] / attempted),
            "adv_pass_rate": float(self.stats["adv_pass"] / attempted),
            "accepted_rate": float(self.stats["accepted"] / attempted),
            "mean_rel_eog": float(np.mean(self.stats["mean_rel_eog"])) if self.stats["mean_rel_eog"] else 0.0,
            "mean_rel_eeg": float(np.mean(self.stats["mean_rel_eeg"])) if self.stats["mean_rel_eeg"] else 0.0,
            "mean_feature_shift": float(np.mean(self.stats["mean_feature_shift"])) if self.stats["mean_feature_shift"] else 0.0,
            "mean_proxy_conflict": float(np.mean(self.stats["mean_proxy_conflict"])) if self.stats["mean_proxy_conflict"] else 0.0,
            "mean_proxy_update_norm": float(np.mean(self.stats["mean_proxy_update_norm"])) if self.stats["mean_proxy_update_norm"] else 0.0,
            "mean_proxy_old_norm": float(np.mean(self.stats["mean_proxy_old_norm"])) if self.stats["mean_proxy_old_norm"] else 0.0,
            "mean_proxy_confidence": float(np.mean(self.stats["mean_proxy_confidence"])) if self.stats["mean_proxy_confidence"] else 0.0,
        }
        self.stats = self.empty_stats()
        return payload


def target_for_attack(logits, attack_mode, state):
    if "ble" in attack_mode:
        return state.ble_target(logits)
    return state.nhe_target(logits)


def is_input_attack(attack_mode):
    return attack_mode.startswith("pgd") or attack_mode.startswith("stealth_") or attack_mode.startswith("proxy_")


def sequence_pass_mask(logits, args):
    probs = logits.softmax(dim=1)
    pred_prob = probs.max(dim=1).values
    return (pred_prob >= args.confidence).sum(dim=1) >= args.confident_epoch_n


def bias_labels(labels, scores, rate, mode, class_num):
    labels = labels.long().clone()
    if rate <= 0 or labels.numel() == 0:
        return labels
    mask = torch.rand(labels.shape, device=labels.device) < rate
    if mode == "next":
        replacement = (labels + 1) % class_num
    elif scores.dim() == 2:
        replacement = scores.detach().topk(2, dim=1).indices[:, 1].reshape_as(labels)
    elif scores.dim() == 3:
        replacement = scores.detach().topk(2, dim=1).indices[:, 1, :].reshape_as(labels)
    else:
        raise ValueError(f"Unsupported score shape for second-best bias: {tuple(scores.shape)}")
    return torch.where(mask, replacement.long(), labels)


def raw_stats(eog, eeg):
    x = torch.cat((eog, eeg), dim=2)
    mean = x.mean(dim=(1, 3))
    std = x.std(dim=(1, 3))
    q05 = torch.quantile(x, 0.05, dim=3).mean(dim=1)
    q95 = torch.quantile(x, 0.95, dim=3).mean(dim=1)
    return torch.cat((mean, std, q05, q95), dim=1)


def poison_stealth_drift_batch(eog_new, eeg_new, blocks, state, args):
    set_train(blocks, False)
    eog_base = eog_new.detach()
    eeg_base = eeg_new.detach()
    eps_eog = eog_base.detach().std().clamp_min(1e-6) * args.stealth_eps_scale
    eps_eeg = eeg_base.detach().std().clamp_min(1e-6) * args.stealth_eps_scale
    step_eog = eps_eog / max(args.stealth_steps // 2, 1)
    step_eeg = eps_eeg / max(args.stealth_steps // 2, 1)

    with torch.no_grad():
        logits_clean_seq = forward_blocks(blocks, eog_base, eeg_base, args)
        logits_clean = flat_logits(logits_clean_seq)
        clean_probs = logits_clean.softmax(dim=1)
        pseudo_labels = clean_probs.argmax(dim=1)
        target_labels = clean_probs.topk(2, dim=1).indices[:, 1]
        clean_pass = sequence_pass_mask(logits_clean_seq, args)
        feat_clean = feature_embeddings(blocks, eog_base, eeg_base, args)
        stats_clean = raw_stats(eog_base, eeg_base)
        direction = state.get_drift_direction(feat_clean.shape[1])

    delta_eog = torch.zeros_like(eog_base)
    delta_eeg = torch.zeros_like(eeg_base)
    if args.pgd_random_start:
        delta_eog.uniform_(-float(eps_eog), float(eps_eog))
        delta_eeg.uniform_(-float(eps_eeg), float(eps_eeg))

    for _ in range(args.stealth_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        eog_adv = eog_base + delta_eog
        eeg_adv = eeg_base + delta_eeg
        logits_adv_seq = forward_blocks(blocks, eog_adv, eeg_adv, args)
        logits_adv = flat_logits(logits_adv_seq)
        probs_adv = logits_adv.softmax(dim=1)
        max_prob_adv = probs_adv.max(dim=1).values
        feat_adv = feature_embeddings(blocks, eog_adv, eeg_adv, args)
        stats_adv = raw_stats(eog_adv, eeg_adv)

        preserve_loss = F.cross_entropy(logits_adv, pseudo_labels)
        pass_loss = F.relu(args.confidence - max_prob_adv).mean()
        raw_loss = F.mse_loss(stats_adv, stats_clean)
        l2_loss = delta_eog.pow(2).mean() / eps_eog.pow(2).clamp_min(1e-12) + delta_eeg.pow(2).mean() / eps_eeg.pow(2).clamp_min(1e-12)
        if args.attack_mode == "stealth_target_flip":
            target_loss = F.cross_entropy(logits_adv, target_labels)
            loss = (
                args.stealth_conf_weight * target_loss
                + args.stealth_pass_weight * pass_loss
                + args.stealth_raw_weight * raw_loss
                + args.stealth_l2_weight * l2_loss
            )
        else:
            drift = ((feat_adv - feat_clean) @ direction).mean()
            centroid = feat_adv.mean(dim=0)
            target_centroid = feat_clean.mean(dim=0) + args.stealth_eta * direction
            align_loss = F.mse_loss(centroid, target_centroid)
            loss = (
                args.stealth_conf_weight * preserve_loss
                + args.stealth_pass_weight * pass_loss
                + args.stealth_raw_weight * raw_loss
                + args.stealth_l2_weight * l2_loss
                + args.stealth_align_weight * align_loss
                - args.stealth_drift_weight * drift
            )
        grad_eog, grad_eeg = torch.autograd.grad(loss, [delta_eog, delta_eeg])
        delta_eog = (delta_eog - step_eog * grad_eog.sign()).detach().clamp(-eps_eog, eps_eog)
        delta_eeg = (delta_eeg - step_eeg * grad_eeg.sign()).detach().clamp(-eps_eeg, eps_eeg)

    eog_adv = eog_base + delta_eog
    eeg_adv = eeg_base + delta_eeg
    with torch.no_grad():
        logits_adv_seq = forward_blocks(blocks, eog_adv, eeg_adv, args)
        adv_pass = sequence_pass_mask(logits_adv_seq, args)
        if args.stealth_accept_adv_only:
            accepted = adv_pass
        else:
            accepted = adv_pass & clean_pass
        mask = accepted.view(-1, 1, 1, 1)
        eog_out = torch.where(mask, eog_adv, eog_base)
        eeg_out = torch.where(mask, eeg_adv, eeg_base)
        rel_eog = torch.linalg.norm((eog_out - eog_base).reshape(eog_base.shape[0], -1), dim=1) / (
            torch.linalg.norm(eog_base.reshape(eog_base.shape[0], -1), dim=1) + 1e-12
        )
        rel_eeg = torch.linalg.norm((eeg_out - eeg_base).reshape(eeg_base.shape[0], -1), dim=1) / (
            torch.linalg.norm(eeg_base.reshape(eeg_base.shape[0], -1), dim=1) + 1e-12
        )
        feat_out = feature_embeddings(blocks, eog_out, eeg_out, args)
        feature_shift = torch.linalg.norm(feat_out - feat_clean, dim=1)
        state.record_stealth(clean_pass, adv_pass, accepted, rel_eog, rel_eeg, feature_shift)
    return eog_out.detach(), eeg_out.detach()


def proxy_meta_parameters(blocks, args):
    if args.proxy_meta_param_scope == "classifier":
        modules = [blocks[2]]
    elif args.proxy_meta_param_scope == "encoder_head":
        modules = [blocks[1], blocks[2]]
    elif args.proxy_meta_param_scope == "all":
        modules = [blocks[0], blocks[1], blocks[2]]
    else:
        raise ValueError(f"Unsupported proxy meta parameter scope: {args.proxy_meta_param_scope}")
    params = []
    for module in modules:
        params.extend([param for param in module.parameters() if param.requires_grad])
    return params


def grad_dot_and_norms(update_grads, old_grads, device):
    dot = torch.zeros((), device=device)
    update_sq = torch.zeros((), device=device)
    old_sq = torch.zeros((), device=device)
    for update_grad, old_grad in zip(update_grads, old_grads):
        if update_grad is not None:
            update_sq = update_sq + update_grad.pow(2).sum()
        if old_grad is not None:
            old_sq = old_sq + old_grad.pow(2).sum()
        if update_grad is not None and old_grad is not None:
            dot = dot + (update_grad * old_grad).sum()
    return dot, update_sq.sqrt(), old_sq.sqrt()


def restore_modes(blocks, modes):
    for block, mode in zip(blocks, modes):
        block.train(mode)


def project_relative_l2(delta, base, max_rel):
    if max_rel <= 0:
        return delta
    flat_delta = delta.reshape(delta.shape[0], -1)
    flat_base = base.reshape(base.shape[0], -1)
    delta_norm = torch.linalg.norm(flat_delta, dim=1).clamp_min(1e-12)
    max_norm = torch.linalg.norm(flat_base, dim=1).clamp_min(1e-12) * max_rel
    scale = torch.minimum(torch.ones_like(delta_norm), max_norm / delta_norm)
    return delta * scale.view(-1, 1, 1, 1)


def project_proxy_delta(delta_eog, delta_eeg, eog_base, eeg_base, args):
    delta_eog = project_relative_l2(delta_eog, eog_base, args.proxy_meta_max_rel_eog)
    delta_eeg = project_relative_l2(delta_eeg, eeg_base, args.proxy_meta_max_rel_eeg)
    return delta_eog, delta_eeg


def poison_proxy_meta_conflict_batch(eog_new, eeg_new, eog_replay, eeg_replay, label_replay, student_blocks, teacher_blocks, state, args):
    student_modes = [block.training for block in student_blocks]
    teacher_modes = [block.training for block in teacher_blocks]
    set_train(student_blocks, False)
    set_train(teacher_blocks, False)

    eog_base = eog_new.detach()
    eeg_base = eeg_new.detach()
    eog_replay = eog_replay.detach()
    eeg_replay = eeg_replay.detach()
    label_replay = label_replay.detach()

    eps_eog = eog_base.detach().std().clamp_min(1e-6) * args.proxy_meta_eps_scale
    eps_eeg = eeg_base.detach().std().clamp_min(1e-6) * args.proxy_meta_eps_scale
    step_eog = eps_eog / max(args.proxy_meta_steps // 2, 1)
    step_eeg = eps_eeg / max(args.proxy_meta_steps // 2, 1)

    meta_params = proxy_meta_parameters(student_blocks, args)
    if not meta_params:
        restore_modes(student_blocks, student_modes)
        restore_modes(teacher_blocks, teacher_modes)
        return eog_new, eeg_new

    with torch.no_grad():
        logits_clean_seq = forward_blocks(teacher_blocks, eog_base, eeg_base, args)
        clean_pass = sequence_pass_mask(logits_clean_seq, args)
        feat_clean = feature_embeddings(student_blocks, eog_base, eeg_base, args)
        stats_clean = raw_stats(eog_base, eeg_base)

    old_logits = flat_logits(forward_blocks(student_blocks, eog_replay, eeg_replay, args))
    old_loss = F.cross_entropy(old_logits, flat_labels(label_replay))
    old_grads = torch.autograd.grad(old_loss, meta_params, retain_graph=False, create_graph=False, allow_unused=True)
    old_grads = [None if grad is None else grad.detach() for grad in old_grads]
    _old_dot, _old_update_norm, old_norm_static = grad_dot_and_norms(old_grads, old_grads, args.device)
    if float(old_norm_static.detach().cpu()) <= 1e-12:
        restore_modes(student_blocks, student_modes)
        restore_modes(teacher_blocks, teacher_modes)
        return eog_new, eeg_new

    delta_eog = torch.zeros_like(eog_base)
    delta_eeg = torch.zeros_like(eeg_base)
    if args.pgd_random_start:
        delta_eog.uniform_(-float(eps_eog), float(eps_eog))
        delta_eeg.uniform_(-float(eps_eeg), float(eps_eeg))
        delta_eog, delta_eeg = project_proxy_delta(delta_eog, delta_eeg, eog_base, eeg_base, args)

    last_conflict = torch.zeros((), device=args.device)
    last_update_norm = torch.zeros((), device=args.device)
    last_confidence = torch.zeros(eog_base.shape[0] * args.model_param.SeqLength, device=args.device)

    for _ in range(args.proxy_meta_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        eog_adv = eog_base + delta_eog
        eeg_adv = eeg_base + delta_eeg

        teacher_logits = flat_logits(forward_blocks(teacher_blocks, eog_adv, eeg_adv, args))
        teacher_probs = teacher_logits.softmax(dim=1)
        pseudo_conf, pseudo_labels = teacher_probs.max(dim=1)
        confident = pseudo_conf > args.confidence
        soft_weights = torch.sigmoid((pseudo_conf - args.confidence) / max(args.proxy_meta_conf_tau, 1e-6))
        update_weights = confident.float() + args.proxy_meta_soft_weight * soft_weights
        update_weights = update_weights.clamp(max=1.0)

        student_logits = flat_logits(forward_blocks(student_blocks, eog_adv, eeg_adv, args))
        ce_new = F.cross_entropy(student_logits, pseudo_labels.detach(), reduction="none")
        if float(update_weights.detach().sum().cpu()) > 0:
            loss_new = (ce_new * update_weights).sum() / update_weights.sum().clamp_min(1e-6)
        else:
            loss_new = ce_new.mean()

        update_grads = torch.autograd.grad(loss_new, meta_params, retain_graph=True, create_graph=True, allow_unused=True)
        dot, update_norm, old_norm = grad_dot_and_norms(update_grads, old_grads, args.device)
        conflict = dot / (update_norm * old_norm + 1e-12)
        confidence_loss = F.relu(args.confidence - pseudo_conf).mean()
        stats_adv = raw_stats(eog_adv, eeg_adv)
        raw_loss = F.mse_loss(stats_adv, stats_clean)
        l2_loss = delta_eog.pow(2).mean() / eps_eog.pow(2).clamp_min(1e-12) + delta_eeg.pow(2).mean() / eps_eeg.pow(2).clamp_min(1e-12)
        loss = (
            args.proxy_meta_conflict_weight * conflict
            + args.proxy_meta_confidence_weight * confidence_loss
            + args.proxy_meta_raw_weight * raw_loss
            + args.proxy_meta_l2_weight * l2_loss
            - args.proxy_meta_grad_norm_weight * update_norm / old_norm_static.clamp_min(1e-12)
        )
        grad_eog, grad_eeg = torch.autograd.grad(loss, [delta_eog, delta_eeg])
        delta_eog = (delta_eog - step_eog * grad_eog.sign()).detach().clamp(-eps_eog, eps_eog)
        delta_eeg = (delta_eeg - step_eeg * grad_eeg.sign()).detach().clamp(-eps_eeg, eps_eeg)
        delta_eog, delta_eeg = project_proxy_delta(delta_eog, delta_eeg, eog_base, eeg_base, args)
        last_conflict = conflict.detach()
        last_update_norm = update_norm.detach()
        last_confidence = pseudo_conf.detach()

    eog_adv = eog_base + delta_eog
    eeg_adv = eeg_base + delta_eeg
    with torch.no_grad():
        logits_adv_seq = forward_blocks(teacher_blocks, eog_adv, eeg_adv, args)
        adv_pass = sequence_pass_mask(logits_adv_seq, args)
        rel_eog = torch.linalg.norm((eog_adv - eog_base).reshape(eog_base.shape[0], -1), dim=1) / (
            torch.linalg.norm(eog_base.reshape(eog_base.shape[0], -1), dim=1) + 1e-12
        )
        rel_eeg = torch.linalg.norm((eeg_adv - eeg_base).reshape(eeg_base.shape[0], -1), dim=1) / (
            torch.linalg.norm(eeg_base.reshape(eeg_base.shape[0], -1), dim=1) + 1e-12
        )
        feat_adv = feature_embeddings(student_blocks, eog_adv, eeg_adv, args)
        feature_shift = torch.linalg.norm(feat_adv - feat_clean, dim=1)
        state.record_proxy_meta(
            clean_pass,
            adv_pass,
            rel_eog,
            rel_eeg,
            feature_shift,
            last_conflict,
            last_update_norm,
            old_norm_static,
            last_confidence,
        )

    restore_modes(student_blocks, student_modes)
    restore_modes(teacher_blocks, teacher_modes)
    return eog_adv.detach(), eeg_adv.detach()


def use_individual_proxy_upload(args):
    return args.attack_mode == "proxy_meta_conflict" and args.proxy_meta_poison_scope == "individual"


def proxy_meta_reference_paths(args):
    if args.proxy_meta_reference == "base_train":
        return args.base_train_paths
    if args.proxy_meta_reference == "target_buffer":
        return args.train_paths
    raise ValueError(f"Unsupported proxy meta reference source: {args.proxy_meta_reference}")


@torch.no_grad()
def proxy_meta_pseudo_labels(blocks, eog, eeg, args):
    modes = [block.training for block in blocks]
    set_train(blocks, False)
    logits = forward_blocks(blocks, eog, eeg, args)
    labels = logits.argmax(dim=1).long()
    restore_modes(blocks, modes)
    return labels


def materialize_proxy_meta_subject(args, subject, num, student_blocks, teacher_blocks, state):
    clean_paths = subject_paths(args.data_root, subject)
    new_loader = DataLoader(SequenceDataset(clean_paths), batch_size=args.batch, shuffle=False, num_workers=args.num_worker)
    replay_loader = DataLoader(SequenceDataset(proxy_meta_reference_paths(args)), batch_size=args.batch, shuffle=True, num_workers=args.num_worker)
    replay_iter = iter(replay_loader)
    save_data_path = args.variant_dir / "poisoned_uploads" / f"individual_{num}" / "data"
    save_data_path.mkdir(parents=True, exist_ok=True)
    total_sequences = len(clean_paths[0])
    poison_fraction = min(max(args.proxy_meta_poison_fraction, 0.0), 1.0)
    poison_count = int(math.ceil(total_sequences * poison_fraction)) if poison_fraction > 0 else 0
    rng = np.random.default_rng(args.seed + int(subject) * 1009 + num)
    poison_indices = set(rng.choice(total_sequences, poison_count, replace=False).tolist()) if poison_count else set()

    poisoned_data_paths = []
    label_paths = []
    global_idx = 0
    materialized_poisoned = 0
    for eog, eeg, _labels in new_loader:
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        try:
            eog_replay, eeg_replay, label_replay = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            eog_replay, eeg_replay, label_replay = next(replay_iter)
        eog_replay = eog_replay.to(args.device)
        eeg_replay = eeg_replay.to(args.device)
        label_replay = label_replay.to(args.device)
        if args.proxy_meta_reference_label_mode == "pseudo":
            label_replay = proxy_meta_pseudo_labels(student_blocks, eog_replay, eeg_replay, args)

        eog_adv, eeg_adv = poison_proxy_meta_conflict_batch(
            eog,
            eeg,
            eog_replay,
            eeg_replay,
            label_replay,
            student_blocks,
            teacher_blocks,
            state,
            args,
        )
        for row in range(eog_adv.shape[0]):
            idx = global_idx + row
            save_data_file = save_data_path / f"{idx}.npy"
            if idx in poison_indices:
                upload_x = torch.cat((eog_adv[row], eeg_adv[row]), dim=1)
                materialized_poisoned += 1
            else:
                upload_x = torch.cat((eog[row], eeg[row]), dim=1)
            np.save(save_data_file, upload_x.detach().cpu().numpy().astype(np.float32))
            poisoned_data_paths.append(save_data_file)
            label_paths.append(clean_paths[1][idx])
        global_idx += eog_adv.shape[0]
    print(
        f"[{args.variant}] Materialized upload for Subject {subject}: "
        f"{materialized_poisoned}/{len(poisoned_data_paths)} poisoned sequences",
        flush=True,
    )
    stats = {
        "step": int(num),
        "subject": int(subject),
        "uploaded": int(len(poisoned_data_paths)),
        "poisoned": int(materialized_poisoned),
        "poison_fraction": float(materialized_poisoned / max(len(poisoned_data_paths), 1)),
        "reference": args.proxy_meta_reference,
        "reference_label_mode": args.proxy_meta_reference_label_mode,
    }
    return (poisoned_data_paths, label_paths), stats


def poison_new_batch(eog_new, eeg_new, blocks, attack_mode, state, args):
    if attack_mode.startswith("stealth_"):
        return poison_stealth_drift_batch(eog_new, eeg_new, blocks, state, args)
    if not attack_mode.startswith("pgd"):
        return eog_new, eeg_new
    set_train(blocks, False)
    eog_base = eog_new.detach()
    eeg_base = eeg_new.detach()
    eps_eog = eog_base.detach().std().clamp_min(1e-6) * args.pgd_eps_scale
    eps_eeg = eeg_base.detach().std().clamp_min(1e-6) * args.pgd_eps_scale
    step_eog = eps_eog / max(args.pgd_steps // 2, 1)
    step_eeg = eps_eeg / max(args.pgd_steps // 2, 1)
    delta_eog = torch.zeros_like(eog_base)
    delta_eeg = torch.zeros_like(eeg_base)
    if args.pgd_random_start:
        delta_eog.uniform_(-float(eps_eog), float(eps_eog))
        delta_eeg.uniform_(-float(eps_eeg), float(eps_eeg))
    for _ in range(args.pgd_steps):
        delta_eog.requires_grad_(True)
        delta_eeg.requires_grad_(True)
        eog_adv = eog_base + delta_eog
        eeg_adv = eeg_base + delta_eeg
        logits = flat_logits(forward_blocks(blocks, eog_adv, eeg_adv, args))
        target = target_for_attack(logits, attack_mode, state)
        loss = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
        grad_eog, grad_eeg = torch.autograd.grad(loss, [delta_eog, delta_eeg])
        # Targeted PGD: descend toward the adversarial target distribution.
        delta_eog = (delta_eog - step_eog * grad_eog.sign()).detach().clamp(-eps_eog, eps_eog)
        delta_eeg = (delta_eeg - step_eeg * grad_eeg.sign()).detach().clamp(-eps_eeg, eps_eeg)
    return eog_base + delta_eog, eeg_base + delta_eeg


def model_attack_step(blocks, eog_new, eeg_new, state, args):
    optimizer = torch.optim.Adam(
        list(blocks[0].parameters()) + list(blocks[1].parameters()) + list(blocks[2].parameters()),
        lr=args.attack_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    set_train(blocks, True)
    logits = flat_logits(forward_blocks(blocks, eog_new, eeg_new, args))
    target = target_for_attack(logits, args.attack_mode, state)
    loss = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def incremental_train(blocks, teacher_blocks, args, new_task_loader, new_task_id, num, attack_state):
    if args.attack_mode == "stealth_source_drift":
        update_source_drift_direction(args, teacher_blocks, new_task_id, attack_state)
    elif args.attack_mode == "stealth_loss_drift":
        update_loss_drift_direction(args, teacher_blocks, attack_state)
    contrastive = CPCProbe(teacher_blocks, args)
    tmp_blocks_teacher = teacher_blocks
    for epoch in range(1, args.ssl_epoch + 1):
        set_train(teacher_blocks, True)
        losses = []
        for eog, eeg, _label in new_task_loader:
            eog, eeg = eog.to(args.device), eeg.to(args.device)
            if is_input_attack(args.attack_mode):
                eog, eeg = poison_new_batch(eog, eeg, teacher_blocks, args.attack_mode, attack_state, args)
            loss, tmp_blocks_teacher = contrastive.update(eeg, eog)
            losses.append(loss)
        print(f"[{args.variant}] Individual {num} Subject {new_task_id} CPC Epoch {epoch} Loss {np.mean(losses):.6f}", flush=True)

    algorithm = BufferPseudoLabelFinetuneProbe(blocks, tmp_blocks_teacher, args)
    buffer_loader = make_new_loader(args, new_task_id, is_buffer=True, shuffle=True)
    optimizer_cea = torch.optim.Adam(
        [{"params": list(blocks[0].parameters())}, {"params": list(blocks[1].parameters())}],
        lr=args.cl_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    stealth_ascent_optimizer = None
    old_proxy_loader = None
    old_proxy_iter = None
    if args.attack_mode.startswith("stealth_") and (
        args.stealth_old_proxy_ascent_weight > 0 or args.stealth_new_entropy_ascent_weight > 0
    ):
        stealth_ascent_optimizer = torch.optim.Adam(
            list(blocks[0].parameters()) + list(blocks[1].parameters()) + list(blocks[2].parameters()),
            lr=args.stealth_ascent_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        if args.stealth_old_proxy_ascent_weight > 0:
            old_proxy_loader = make_loader(args.data_root, sorted(args.old_idx), args.batch, shuffle=True, num_workers=args.num_worker)
            old_proxy_iter = iter(old_proxy_loader)
    align_feature = []
    tmp_blocks = blocks
    for epoch in range(args.incremental_epoch):
        set_train(blocks, True)
        set_train(tmp_blocks_teacher, False)
        losses = []
        if epoch % args.cross_epoch == 0:
            align_feature.append([])
        for batch_idx, (eog, eeg, label) in enumerate(buffer_loader):
            eog, eeg, label = eog.to(args.device), eeg.to(args.device), label.to(args.device)
            if is_input_attack(args.attack_mode) and not use_individual_proxy_upload(args):
                seq_len = args.model_param.SeqLength
                eog_new, eeg_new = eog[:, :seq_len, :, :], eeg[:, :seq_len, :, :]
                if args.attack_mode.startswith("proxy_"):
                    eog_adv, eeg_adv = poison_proxy_meta_conflict_batch(
                        eog_new,
                        eeg_new,
                        eog[:, seq_len:, :, :],
                        eeg[:, seq_len:, :, :],
                        label[:, seq_len:],
                        tmp_blocks,
                        tmp_blocks_teacher,
                        attack_state,
                        args,
                    )
                else:
                    eog_adv, eeg_adv = poison_new_batch(eog_new, eeg_new, tmp_blocks_teacher, args.attack_mode, attack_state, args)
                eog = torch.cat((eog_adv, eog[:, seq_len:, :, :]), dim=1)
                eeg = torch.cat((eeg_adv, eeg[:, seq_len:, :, :]), dim=1)
            loss, tmp_blocks, feature_before = algorithm.update(eeg, eog, label)
            if args.attack_mode.startswith("model_"):
                seq_len = args.model_param.SeqLength
                loss_attack = model_attack_step(tmp_blocks, eog[:, :seq_len, :, :], eeg[:, :seq_len, :, :], attack_state, args)
                loss += loss_attack
            if stealth_ascent_optimizer is not None:
                ascent_terms = []
                set_train(tmp_blocks, True)
                if args.stealth_new_entropy_ascent_weight > 0:
                    seq_len = args.model_param.SeqLength
                    logits_new = flat_logits(forward_blocks(tmp_blocks, eog[:, :seq_len, :, :], eeg[:, :seq_len, :, :], args))
                    log_probs_new = F.log_softmax(logits_new, dim=1)
                    probs_new = log_probs_new.exp()
                    entropy_new = -(probs_new * log_probs_new).sum(dim=1).mean()
                    ascent_terms.append(-args.stealth_new_entropy_ascent_weight * entropy_new)
                if args.stealth_old_proxy_ascent_weight > 0:
                    try:
                        eog_old, eeg_old, label_old = next(old_proxy_iter)
                    except StopIteration:
                        old_proxy_iter = iter(old_proxy_loader)
                        eog_old, eeg_old, label_old = next(old_proxy_iter)
                    eog_old = eog_old.to(args.device)
                    eeg_old = eeg_old.to(args.device)
                    label_old = label_old.to(args.device)
                    logits_old = flat_logits(forward_blocks(tmp_blocks, eog_old, eeg_old, args))
                    old_proxy_loss = F.cross_entropy(logits_old, flat_labels(label_old))
                    ascent_terms.append(-args.stealth_old_proxy_ascent_weight * old_proxy_loss)
                if ascent_terms:
                    ascent_loss = sum(ascent_terms)
                    stealth_ascent_optimizer.zero_grad()
                    ascent_loss.backward()
                    stealth_ascent_optimizer.step()
                    loss += float(ascent_loss.detach().cpu())
            if epoch % args.cross_epoch == 0:
                align_feature[-1].append(feature_before)
            if epoch % args.cross_epoch == 0 and epoch != 0:
                seq_len = args.model_param.SeqLength
                eog_train = eog[:, seq_len:, :, :].contiguous().reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
                eeg_train = eeg[:, seq_len:, :, :].contiguous().reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
                feature_latter = blocks[1](blocks[0](eeg_train, eog_train))
                optimizer_cea.zero_grad()
                z1 = F.log_softmax(feature_latter, dim=-1)
                z2 = F.softmax(align_feature[-2][batch_idx].detach(), dim=-1)
                loss_cea = F.kl_div(z1, z2, reduction="batchmean")
                loss_cea.backward()
                optimizer_cea.step()
            losses.append(loss)
        print(f"[{args.variant}] Individual {num} Subject {new_task_id} Joint Epoch {epoch+1} Loss {np.mean(losses):.6f}", flush=True)
    return tmp_blocks, tmp_blocks_teacher


def buffer_single_merge(args, new_task_id, num, tmp_blocks, attack_state):
    new_paths = get_uploaded_subject_paths(args, new_task_id, use_uploaded=True)
    new_loader = DataLoader(SequenceDataset(new_paths), batch_size=args.batch, shuffle=False, num_workers=args.num_worker)
    save_path = args.variant_dir / "pseudo_labels" / f"individual_{num}" / "label"
    save_data_path = args.variant_dir / "poisoned_sequences" / f"individual_{num}" / "data"
    save_path.mkdir(parents=True, exist_ok=True)
    if args.store_poisoned_buffer and not use_individual_proxy_upload(args):
        save_data_path.mkdir(parents=True, exist_ok=True)
    added = 0
    biased = 0
    global_idx = 0
    set_train(tmp_blocks, False)
    for eog, eeg, _labels in new_loader:
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        if args.store_poisoned_buffer and is_input_attack(args.attack_mode) and not use_individual_proxy_upload(args):
            eog_eval, eeg_eval = poison_new_batch(eog, eeg, tmp_blocks, args.attack_mode, attack_state, args)
        else:
            eog_eval, eeg_eval = eog, eeg
        with torch.no_grad():
            logits = forward_blocks(tmp_blocks, eog_eval, eeg_eval, args)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_prob = probs.max(axis=1)
        pred_label = probs.argmax(axis=1)
        for row in range(pred_prob.shape[0]):
            idx = global_idx + row
            confident_epoch_num = np.sum(pred_prob[row, :] >= args.confidence)
            if confident_epoch_num >= args.confident_epoch_n:
                save_label_path = save_path / f"{idx}.npy"
                label_to_save = np.squeeze(pred_label[row, :].reshape(-1, 1))
                if args.attack_mode.startswith("stealth_") and args.stealth_buffer_bias_rate > 0:
                    if np.random.random() < args.stealth_buffer_bias_rate:
                        if args.stealth_buffer_bias_mode == "next":
                            label_to_save = (label_to_save + 1) % args.model_param.NumClasses
                        else:
                            label_to_save = np.argsort(probs[row, :, :], axis=0)[-2, :].reshape(-1)
                        biased += 1
                np.save(save_label_path, label_to_save)
                if args.store_poisoned_buffer and is_input_attack(args.attack_mode) and not use_individual_proxy_upload(args):
                    save_data_file = save_data_path / f"{idx}.npy"
                    poisoned_x = torch.cat((eog_eval[row], eeg_eval[row]), dim=1).detach().cpu().numpy().astype(np.float32)
                    np.save(save_data_file, poisoned_x)
                    args.train_paths[0].append(save_data_file)
                else:
                    args.train_paths[0].append(new_paths[0][idx])
                args.train_paths[1].append(save_label_path)
                added += 1
        global_idx += pred_prob.shape[0]
    return added, biased


def summarize_plasticity(performance):
    acc_initial, acc_before, acc_after = [], [], []
    mf1_initial, mf1_before, mf1_after = [], [], []
    for metrics in performance["plasticity"].values():
        if len(metrics["ACC"]) < 3 or len(metrics["MF1"]) < 3:
            continue
        acc_initial.append(metrics["ACC"][0])
        acc_before.append(metrics["ACC"][1])
        acc_after.append(metrics["ACC"][2])
        mf1_initial.append(metrics["MF1"][0])
        mf1_before.append(metrics["MF1"][1])
        mf1_after.append(metrics["MF1"][2])
    return {
        "initial_acc": float(np.mean(acc_initial)) if acc_initial else math.nan,
        "before_acc": float(np.mean(acc_before)) if acc_before else math.nan,
        "after_acc": float(np.mean(acc_after)) if acc_after else math.nan,
        "initial_mf1": float(np.mean(mf1_initial)) if mf1_initial else math.nan,
        "before_mf1": float(np.mean(mf1_before)) if mf1_before else math.nan,
        "after_mf1": float(np.mean(mf1_after)) if mf1_after else math.nan,
    }


def write_progress(args, performance, summary=None):
    payload = {"performance": performance, "summary": summary or summarize_plasticity(performance)}
    path = args.variant_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_variant(base_args, variant, attack_mode, new_order):
    args = copy.copy(base_args)
    args.variant = variant
    args.attack_mode = attack_mode
    args.variant_dir = args.output_root / variant
    args.variant_dir.mkdir(parents=True, exist_ok=True)
    args.train_paths = [list(args.base_train_paths[0]), list(args.base_train_paths[1])]
    args.train_len = len(args.train_paths[0])
    args.alpha = args.initial_alpha
    args.uploaded_subject_paths = {}

    blocks = load_pretrained(args)
    initial_blocks = load_pretrained(args)
    old_loader = make_loader(args.data_root, sorted(args.old_idx), args.batch, shuffle=False, num_workers=args.num_worker)
    performance = {
        "stability": {"ACC": [], "MF1": [], "AAA": [], "AAF1": [], "FR": []},
        "plasticity": {str(int(subject)): {"ACC": [], "MF1": []} for subject in new_order},
        "buffer": [],
        "attack_diagnostics": [],
        "uploaded_subjects": [],
        "order": [int(subject) for subject in new_order],
        "attack_mode": attack_mode,
    }
    attack_state = AttackState(args.model_param.NumClasses, args.device)

    old_result = evaluate(blocks, old_loader, args)
    performance["stability"]["ACC"].append(old_result["acc"])
    performance["stability"]["MF1"].append(old_result["mf1"])
    performance["stability"]["AAA"].append(float(compute_aaa(performance["stability"]["ACC"])))
    performance["stability"]["AAF1"] = compute_aaf1(performance["stability"]["MF1"])
    performance["stability"]["FR"].append(float(compute_forget(performance["stability"]["ACC"])))
    if not args.no_save_checkpoints:
        save_blocks(blocks, args.variant_dir / "checkpoints" / "Pretrain", args.seed)

    for num, subject in enumerate(new_order, start=1):
        if num >= args.train_num:
            args.alpha = np.power(0.1, np.log10(num / args.train_num) + 2)
        print(f"[{variant}] New Task {num}/{len(new_order)} Subject {subject} attack={attack_mode}", flush=True)
        last_blocks = clone_blocks(blocks, args)
        teacher_blocks = clone_blocks(blocks, args)
        if use_individual_proxy_upload(args):
            upload_paths, upload_stats = materialize_proxy_meta_subject(
                args,
                subject,
                num,
                blocks,
                teacher_blocks,
                attack_state,
            )
            args.uploaded_subject_paths[int(subject)] = upload_paths
            performance["uploaded_subjects"].append(upload_stats)
        new_loader = make_new_loader(args, subject, is_buffer=False, shuffle=True, use_uploaded=True)
        blocks, teacher_blocks = incremental_train(blocks, teacher_blocks, args, new_loader, subject, num, attack_state)
        if not args.no_save_checkpoints:
            save_blocks(blocks, args.variant_dir / "checkpoints" / f"individual_{num}", args.seed)

        new_eval_loader = make_new_loader(args, subject, is_buffer=False, shuffle=False, use_uploaded=False)
        initial_result = evaluate(initial_blocks, new_eval_loader, args)
        before_result = evaluate(last_blocks, new_eval_loader, args)
        after_result = evaluate(blocks, new_eval_loader, args)
        performance["plasticity"][str(int(subject))]["ACC"] = [initial_result["acc"], before_result["acc"], after_result["acc"]]
        performance["plasticity"][str(int(subject))]["MF1"] = [initial_result["mf1"], before_result["mf1"], after_result["mf1"]]

        old_result = evaluate(blocks, old_loader, args)
        performance["stability"]["ACC"].append(old_result["acc"])
        performance["stability"]["MF1"].append(old_result["mf1"])
        performance["stability"]["AAA"].append(float(compute_aaa(performance["stability"]["ACC"])))
        performance["stability"]["AAF1"] = compute_aaf1(performance["stability"]["MF1"])
        performance["stability"]["FR"].append(float(compute_forget(performance["stability"]["ACC"])))
        added, biased = buffer_single_merge(args, subject, num, blocks, attack_state)
        buffer_row = {"step": num, "subject": int(subject), "added": int(added), "length": len(args.train_paths[0])}
        if biased:
            buffer_row["biased"] = int(biased)
        performance["buffer"].append(buffer_row)
        if attack_mode.startswith("stealth_") or attack_mode.startswith("proxy_"):
            attack_stats = attack_state.flush_stats()
            attack_stats.update({"step": num, "subject": int(subject)})
            performance["attack_diagnostics"].append(attack_stats)
        write_progress(args, performance)
        print(
            f"[{variant}] Subject {subject} old ACC={old_result['acc']:.4f} MF1={old_result['mf1']:.4f} "
            f"buffer={len(args.train_paths[0])} added={added}",
            flush=True,
        )
    summary = summarize_plasticity(performance)
    write_progress(args, performance, summary)
    return performance, summary


def write_comparison(output_root, clean_perf, attack_perf, clean_summary, attack_summary, args):
    report = {
        "config": {
            "seed": args.seed,
            "data_root": str(args.data_root),
            "input_checkpoint_root": str(args.input_checkpoint_root),
            "new_order": [int(x) for x in args.new_order],
            "max_subjects": args.max_subjects,
            "ssl_epoch": args.ssl_epoch,
            "incremental_epoch": args.incremental_epoch,
            "attack_mode": args.attack_mode,
            "proxy_meta_poison_scope": getattr(args, "proxy_meta_poison_scope", "batch"),
            "proxy_meta_poison_fraction": getattr(args, "proxy_meta_poison_fraction", 1.0),
            "proxy_meta_reference": getattr(args, "proxy_meta_reference", "target_buffer"),
            "proxy_meta_reference_label_mode": getattr(args, "proxy_meta_reference_label_mode", "true"),
            "proxy_meta_max_rel_eog": getattr(args, "proxy_meta_max_rel_eog", 0.0),
            "proxy_meta_max_rel_eeg": getattr(args, "proxy_meta_max_rel_eeg", 0.0),
        },
        "clean_summary": clean_summary,
        "attack_summary": attack_summary,
        "clean_final": {
            "acc": clean_perf["stability"]["ACC"][-1],
            "mf1": clean_perf["stability"]["MF1"][-1],
            "aaa": clean_perf["stability"]["AAA"][-1],
            "aaf1": clean_perf["stability"]["AAF1"][-1],
            "fr": clean_perf["stability"]["FR"][-1],
        },
        "attack_final": {
            "acc": attack_perf["stability"]["ACC"][-1],
            "mf1": attack_perf["stability"]["MF1"][-1],
            "aaa": attack_perf["stability"]["AAA"][-1],
            "aaf1": attack_perf["stability"]["AAF1"][-1],
            "fr": attack_perf["stability"]["FR"][-1],
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["# BrainUICL RTTDP-style CL Comparison\n\n"]
    lines.append("This run uses the same new-individual order for clean and attacked variants.\n\n")
    lines.append("## Config\n\n```json\n" + json.dumps(report["config"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("## Final Stability\n\n")
    lines.append("| variant | ACC | MF1 | AAA | AAF1 | FR |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, final in [("clean", report["clean_final"]), ("attack", report["attack_final"])]:
        lines.append(
            f"| {name} | {final['acc']:.4f} | {final['mf1']:.4f} | {final['aaa']:.4f} | "
            f"{final['aaf1']:.4f} | {final['fr']:.4f} |\n"
        )
    lines.append("\n## Final Plasticity\n\n")
    lines.append("| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, summary in [("clean", clean_summary), ("attack", attack_summary)]:
        lines.append(
            f"| {name} | {summary['initial_acc']:.4f} | {summary['before_acc']:.4f} | {summary['after_acc']:.4f} | "
            f"{summary['initial_mf1']:.4f} | {summary['before_mf1']:.4f} | {summary['after_mf1']:.4f} |\n"
        )
    lines.append("\n## Stability Curves\n\n")
    lines.append("```text\n")
    lines.append(f"clean ACC:  {clean_perf['stability']['ACC']}\n")
    lines.append(f"attack ACC: {attack_perf['stability']['ACC']}\n")
    lines.append(f"clean MF1:  {clean_perf['stability']['MF1']}\n")
    lines.append(f"attack MF1: {attack_perf['stability']['MF1']}\n")
    lines.append("```\n")
    (output_root / "comparison_report.md").write_text("".join(lines))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--input-checkpoint-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "rttdp_brainuicl_runs" / "latest")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0, help="0 means all new individuals.")
    parser.add_argument("--ssl-epoch", type=int, default=10)
    parser.add_argument("--incremental-epoch", type=int, default=10)
    parser.add_argument("--cross-epoch", type=int, default=2)
    parser.add_argument(
        "--attack-mode",
        choices=[
            "model_nhe",
            "model_ble",
            "pgd_nhe",
            "pgd_ble",
            "stealth_drift",
            "stealth_source_drift",
            "stealth_loss_drift",
            "stealth_target_flip",
            "proxy_meta_conflict",
        ],
        default="model_nhe",
    )
    parser.add_argument("--run-clean-only", action="store_true")
    parser.add_argument("--run-attack-only", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--cl-lr", type=float, default=1e-7)
    parser.add_argument("--attack-lr", type=float, default=8e-5)
    parser.add_argument("--initial-alpha", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--confident-epoch-n", type=int, default=15)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--pgd-eps-scale", type=float, default=0.10)
    parser.add_argument("--pgd-random-start", action="store_true")
    parser.add_argument("--stealth-steps", type=int, default=5)
    parser.add_argument("--stealth-eps-scale", type=float, default=0.01)
    parser.add_argument("--stealth-eta", type=float, default=0.05)
    parser.add_argument("--stealth-conf-weight", type=float, default=1.0)
    parser.add_argument("--stealth-pass-weight", type=float, default=2.0)
    parser.add_argument("--stealth-raw-weight", type=float, default=0.05)
    parser.add_argument("--stealth-l2-weight", type=float, default=0.01)
    parser.add_argument("--stealth-align-weight", type=float, default=0.1)
    parser.add_argument("--stealth-drift-weight", type=float, default=0.2)
    parser.add_argument("--stealth-centroid-batches", type=int, default=4)
    parser.add_argument("--stealth-direction-momentum", type=float, default=0.7)
    parser.add_argument("--stealth-accept-adv-only", action="store_true")
    parser.add_argument("--stealth-buffer-bias-rate", type=float, default=0.0)
    parser.add_argument("--stealth-buffer-bias-mode", choices=["second", "next"], default="second")
    parser.add_argument("--stealth-train-new-bias-rate", type=float, default=0.0)
    parser.add_argument("--stealth-train-replay-bias-rate", type=float, default=0.0)
    parser.add_argument("--stealth-train-bias-mode", choices=["second", "next"], default="second")
    parser.add_argument("--stealth-train-new-loss-scale", type=float, default=1.0)
    parser.add_argument("--stealth-new-ascent-weight", type=float, default=0.0)
    parser.add_argument("--stealth-replay-ascent-weight", type=float, default=0.0)
    parser.add_argument("--stealth-ascent-lr", type=float, default=1e-7)
    parser.add_argument("--stealth-old-proxy-ascent-weight", type=float, default=0.0)
    parser.add_argument("--stealth-new-entropy-ascent-weight", type=float, default=0.0)
    parser.add_argument("--proxy-meta-steps", type=int, default=3)
    parser.add_argument("--proxy-meta-eps-scale", type=float, default=0.20)
    parser.add_argument("--proxy-meta-poison-scope", choices=["batch", "individual"], default="batch")
    parser.add_argument("--proxy-meta-poison-fraction", type=float, default=1.0)
    parser.add_argument("--proxy-meta-reference", choices=["target_buffer", "base_train"], default="target_buffer")
    parser.add_argument("--proxy-meta-reference-label-mode", choices=["true", "pseudo"], default="true")
    parser.add_argument("--proxy-meta-max-rel-eog", type=float, default=0.0)
    parser.add_argument("--proxy-meta-max-rel-eeg", type=float, default=0.0)
    parser.add_argument("--proxy-meta-param-scope", choices=["classifier", "encoder_head", "all"], default="classifier")
    parser.add_argument("--proxy-meta-conflict-weight", type=float, default=1.0)
    parser.add_argument("--proxy-meta-confidence-weight", type=float, default=0.5)
    parser.add_argument("--proxy-meta-raw-weight", type=float, default=0.005)
    parser.add_argument("--proxy-meta-l2-weight", type=float, default=0.001)
    parser.add_argument("--proxy-meta-grad-norm-weight", type=float, default=0.1)
    parser.add_argument("--proxy-meta-conf-tau", type=float, default=0.05)
    parser.add_argument("--proxy-meta-soft-weight", type=float, default=0.25)
    parser.add_argument("--store-poisoned-buffer", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fix_randomness(args.seed)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args.dataset = "ISRUC"
    args.algorithm = "cpc"
    args.model_param = ModelConfig(args.dataset)
    args.alpha = args.initial_alpha

    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    args.train_num = len(train_idx)
    args.old_idx = old_idx
    args.base_train_paths = merge_subject_paths(args.data_root, list(train_idx))
    new_order = list(new_idx)
    if args.max_subjects:
        new_order = new_order[: args.max_subjects]
    args.new_order = new_order
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "train": sorted(int(x) for x in train_idx),
        "val": sorted(int(x) for x in val_idx),
        "old_generalization": sorted(int(x) for x in old_idx),
        "new_order": [int(x) for x in new_order],
        "full_new_order": [int(x) for x in new_idx],
    }
    (args.output_root / "split.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=False))
    print(json.dumps({"device": str(args.device), **split_payload}, indent=2), flush=True)

    clean_perf = attack_perf = None
    clean_summary = attack_summary = None
    if not args.run_attack_only:
        clean_perf, clean_summary = run_variant(args, "clean", "none", new_order)
    if not args.run_clean_only:
        attack_perf, attack_summary = run_variant(args, f"attack_{args.attack_mode}", args.attack_mode, new_order)
    if clean_perf is not None and attack_perf is not None:
        report = write_comparison(args.output_root, clean_perf, attack_perf, clean_summary, attack_summary, args)
        print(json.dumps(report["clean_final"], indent=2), flush=True)
        print(json.dumps(report["attack_final"], indent=2), flush=True)


if __name__ == "__main__":
    main()
