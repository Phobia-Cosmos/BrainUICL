# SPR-EEG Continual Learning Defense Report

Date: 2026-07-11

This report adapts Self-Purified Replay (SPR, ICCV 2021) to the
BrainUICL ISRUC individual continual-learning protocol and evaluates both its
useful operating range and its failure modes.

## 1. Experimental Scope

- Dataset: ISRUC subgroup I, 98 available subjects.
- Input: sequences of 20 sleep epochs, each with 2 EOG + 6 EEG channels.
- Task: five-class sleep staging.
- Split seed: 4321.
- Probe stream: the first 10 BrainUICL new individuals.
- Training budget: 3 CPC epochs and 3 joint incremental epochs per individual.
- Pretrained model and source replay data are identical across variants.
- Metrics: final old/generalization ACC, MF1, average accuracy (AAA), average
  MF1 (AAF1), forgetting rate (FR), new-individual plasticity, and replay-label
  error measured with held-out ground truth.

Ground-truth labels are only used to report replay-label error. The defense
does not read them.

## 2. Extracted SPR Method

The original image SPR method contains two networks and two memories:

1. Delayed buffer `D`: temporarily holds the current data stream.
2. Purified buffer `P`: stores samples judged likely to have clean labels.
3. Expert network: trained self-supervised on `D`, then used to compute
   class-conditional feature centrality.
4. Base network: trained with self-supervised replay on `D union P`.
5. Self-Centered Filter: constructs one feature-similarity graph per observed
   class, estimates stochastic eigenvector centrality, fits a two-component
   Beta mixture, and interprets the high-centrality posterior as cleanliness.
6. Downstream inference: supervised training only uses purified memory.

## 3. EEG Mapping

| SPR component | BrainUICL / EEG implementation |
| --- | --- |
| Incoming task-free stream | Sequential unseen ISRUC individuals |
| Delayed buffer | All sequences from the current individual |
| Observed noisy label | BrainUICL teacher pseudo-label for each sleep epoch |
| Expert self-supervision | BrainUICL CPC adapted only on the current individual |
| Base Self-Replay | Optional CPC on current data plus sampled replay data |
| Class graph vertex | One 30-second EEG epoch embedding |
| Class graph grouping | Predicted sleep-stage pseudo-label |
| Edge weight | Non-negative cosine similarity between expert embeddings |
| Stochastic ensemble | Five sampled similarity graphs per sleep stage |
| Clean posterior | Two-component Beta-mixture posterior over centrality |
| Purified memory unit | A 20-epoch EEG sequence |

Sequence acceptance first requires BrainUICL's original confidence rule
(`15/20` epochs with confidence at least `0.9`). The epoch clean posteriors are
then aggregated into a sequence score. A ranked minimum-acceptance fallback
keeps the highest-centrality 75% of candidates when an absolute threshold
would remove too much individual or class coverage.

## 4. Implementation

- `model/spr_eeg.py`
  - stochastic graph construction;
  - power-iteration eigenvector centrality;
  - guarded two-component Beta-mixture EM;
  - epoch-to-sequence purification.
- `experiments/rttdp_brainuicl_full.py`
  - `--defense-mode spr`;
  - optional EEG Self-Replay;
  - SPR buffer filtering and purity diagnostics;
  - reproducible symmetric buffer-label noise;
  - clean, noisy, and adaptive-attack variants with reset random seeds.
- `tests/test_spr_eeg.py`
  - verifies that pseudo-label/feature-cluster mismatches receive lower clean
    probabilities;
  - verifies input-shape validation.

## 5. Main Results

### 5.1 Clean stream and 40% random buffer-label noise

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| SPR ranked clean | 0.7005 | 0.6666 | 0.7059 | 0.6841 | 0.0028 |
| BrainUICL + 40% noise | 0.7005 | 0.6734 | 0.7091 | 0.6886 | 0.0028 |
| Full SPR, strict filter | 0.6861 | 0.6515 | 0.7029 | 0.6817 | 0.0233 |
| Full SPR, relaxed filter | 0.6775 | 0.6416 | 0.6923 | 0.6661 | 0.0355 |
| SPR ranked filter-only | **0.7059** | **0.6805** | 0.6980 | 0.6761 | 0.0049 |

The ranked filter-only variant improves final noisy-stream ACC by 0.54
percentage points and MF1 by 0.71 points over noisy BrainUICL. It also has no
measurable clean-stream penalty in this probe.

However, the final gain is small and AAA/AAF1 are lower. The defense changes
the intermediate trajectory and does not dominate BrainUICL at every step.

### 5.2 Purification diagnostics

| Variant | Mean error before | Mean error after | Accepted / candidates |
| --- | ---: | ---: | ---: |
| Strict SPR | 0.5569 | **0.4339** | 93 / 191 |
| Ranked SPR | 0.5304 | 0.5196 | 172 / 227 |

Strict filtering removes substantially more noisy labels, but loses too much
EEG individual/class coverage and hurts classification. Ranked filtering gives
up most of the purity gain to preserve diversity, producing better final
accuracy. This is the central purity-diversity tradeoff on EEG.

### 5.3 Adaptive proxy-meta poisoning

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| BrainUICL proxy-meta | 0.6195 | 0.5684 | 0.5774 | 0.5146 | 0.1181 |
| SPR ranked proxy-meta | **0.5495** | **0.4874** | 0.5749 | 0.5130 | 0.2178 |

SPR does not defend this attack. Mean pseudo-label error changes from 0.7432
before filtering to 0.7464 after filtering. The attacker moves many samples
into a coherent, high-confidence wrong cluster, violating SPR's assumption
that clean samples form the largest central feature cluster inside each label.
Filtering then retains the attack cluster and removes useful diversity.

The stronger direct `model_nhe` diagnostic similarly collapses SPR ranked to
ACC 0.2270 and FR 0.6768. A replay-purification defense is not expected to stop
an attacker that directly changes model updates.

## 6. Interpretation

The extracted method is useful as a narrow label-noise defense:

- It identifies isolated pseudo-label/feature mismatches.
- It can measurably increase replay purity.
- With diversity-preserving ranked selection, it provides a small final gain
  under random buffer-label noise without harming clean final accuracy.

It is not a general poisoning defense:

- Full Self-Replay directly transferred from image SPR causes EEG feature
  drift under this short BrainUICL budget.
- Absolute purity thresholds remove too many subject-specific sequences.
- Centrality cannot identify a coherent adversarial cluster whose labels,
  confidence, and features have all moved together.

The practical configuration from this probe is therefore `SPR ranked
filter-only`, not the literal full image SPR recipe. A stronger EEG defense
would need temporal consistency, source-anchor distances, class/subject quotas,
and an explicit detector for coherent distribution shifts.

## 7. Reproduction Commands

Environment:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python
```

BrainUICL clean/noisy plus SPR noisy:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --no-save-checkpoints
```

Ranked SPR filter-only:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --run-defense-only \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

Proxy-meta comparison:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --attack-mode proxy_meta_conflict \
  --proxy-meta-poison-scope individual --proxy-meta-steps 5 \
  --proxy-meta-eps-scale 0.50 --proxy-meta-param-scope classifier \
  --proxy-meta-conflict-weight 5.0 --proxy-meta-confidence-weight 0.1 \
  --proxy-meta-grad-norm-weight 0.0 --proxy-meta-raw-weight 0.001 \
  --proxy-meta-l2-weight 0.0005 --pgd-random-start \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

## 8. Result Locations

- `experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_clean_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321`

