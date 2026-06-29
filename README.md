# TransfQMix — TDOA Multi-Agent Tracking

Transformer-based QMIX for cooperative multi-agent reinforcement learning applied to underwater TDOA (Time Difference of Arrival) passive acoustic target tracking.

Reference: Clementi, L. (2026). *Multi-Agent Reinforcement Learning for Underwater TDOA Tracking via Autonomous Vehicles*. Master's thesis, Universitat Pompeu Fabra.

## Setup

```bash
pip install -r requirements_vast.txt
```

Python 3.8 required. GPU recommended (CUDA-capable, 12GB+ VRAM).

## Quick start

Train a 3-agent 1-target TDOA tracking policy:

```bash
# Oracle baseline (ground-truth target positions)
python src/main.py --config=transf_qmix --env-config=mpe/tdoa_tracking_static_minimal with seed=42

# TDOA pipeline (least-squares solver estimates, σ=0.05)
python src/main.py --config=transf_qmix --env-config=mpe/tdoa_tracking_static_minimal_b1 with seed=42

# Range-Only (zero-noise ranges, hidden target positions)
python src/main.py --config=transf_qmix --env-config=mpe/tdoa_tracking_static_minimal with seed=42 \
  env_args.target_pos_mode=hidden env_args.noise_std=0.0 env_args.hide_target_positions=True

# Moving targets
python src/main.py --config=transf_qmix --env-config=mpe/tdoa_tracking_moving_b1 with seed=42
```

## Evaluation

```bash
# Evaluate a trained checkpoint
python src/main.py --config=transf_qmix --env-config=mpe/tdoa_tracking_static_minimal_b1 \
  with evaluate=True test_nepisode=100 env_args.target_pos_mode=tdoa_estimate \
  checkpoint_path=<path> load_step=500100
```

## Key configs

| Config | Description |
|--------|-------------|
| `transf_qmix` | Main 3v1 algorithm config |
| `transf_qmix_6vs2` | 6-agent 2-target variant |
| `tdoa_tracking_static_minimal` | Oracle baseline env (3v1) |
| `tdoa_tracking_static_minimal_b1` | TDOA pipeline env (3v1, σ=0.05) |
| `tdoa_tracking_moving_b1` | TDOA pipeline with moving targets |

## Observation modes

Set via `env_args.target_pos_mode`:
- `ground_truth` — Oracle: exact target positions
- `tdoa_estimate` — TDOA: least-squares solver position estimates
- `hidden` — Range-Only: raw noisy ranges, target position zeroed (use `env_args.hide_target_positions=True`)

## Architecture

TransfQMix (Gallici et. al, 2024)

## Directory structure

```
src/
├── main.py              # Entry point
├── config/              # YAML configs (algs + envs)
├── run/                 # Training run orchestration
├── learners/            # RL algorithm implementations
├── modules/agents/      # Agent network architectures
├── modules/mixers/      # Mixing network architectures
├── modules/layer/       # Transformer layers
├── controllers/         # Multi-agent controllers (MACs)
├── components/          # Replay buffer, epsilon schedules
├── runners/             # Episode collection runners
├── envs/mpe/            # Multi-Particle Environment + TDOA scenarios
└── utils/               # RL utilities, logging
```
