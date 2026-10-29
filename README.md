# RAI pretrained version

This is the basic version of the RAI architecture. The vision encoder and the vision-language model start from pretrained weights and are fine-tuned. The remaining modules are trained on a laser-engraver manipulation task with a Genesis robot, using either a two-finger gripper or a Wuji Hand 2.

The repository is organized by branch:

| Branch | Contents |
| --- | --- |
| ForceVLA | Original ForceVLA. This branch. |
| ForceVLA2 | Reimplementation of ForceVLA2. Coming soon. |
| RAI pretrained | Extended version of ForceVLA2. Coming soon. |

Training data: link coming soon.

## Requirements

Ubuntu 22.04, Python 3.11, and an NVIDIA GPU. Estimates below are for a single GPU. Set `fsdp_devices` in the training config to shard across several GPUs. Multi-node training is not supported.

| Mode | Memory | Example GPU |
| --- | --- | --- |
| Inference | > 8 GB | RTX 4090 |
| Fine-tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-tuning (full) | > 70 GB | A100 80GB / H100 |

## Installation

From the repository root:

```bash
conda create -n rai python=3.11 -y
conda activate rai
python -m pip install --upgrade pip setuptools wheel
conda install -c nvidia cuda-toolkit=12.8 -y
conda install -c conda-forge ffmpeg=7.1.1 -y

pip install -e ./lerobot
pip install -e .
pip install -e ./dlimp
pip install -e ./packages/openpi-client
pip install -e ./flaxformer
```

Point LeRobot at the local dataset root before computing stats or training:

```bash
export HF_LEROBOT_HOME="/path/to/datasets"
```

## Training

Two configs are available:

- `forcevla_lora` fine-tunes with LoRA (`gemma_2b_lora` for the VLM, `gemma_300m_lora` for the action expert).
- `forcevla_full` fine-tunes the full model.

`--task` selects the data. Pass one task name, or `all` to train on every task. Omitting `--task` uses the config default, `flexiv_peel_cucumber_inputForce`.

Task names:

- `flexiv_1plug_insert_inputForce`
- `flexiv_1plug_insert_noForce`
- `flexiv_insert_USB_inputForce`
- `flexiv_insert_USB_noForce`
- `flexiv_peel_cucumber_inputForce`
- `flexiv_peel_cucumber_noForce`
- `flexiv_pump_1bottle_inputForce`
- `flexiv_pump_1bottle_noForce`
- `flexiv_wipe_board_inputForce`
- `flexiv_wipe_board_noForce`
- `all`

Compute normalization stats once for the same config and task you will train. Stats for `all` are stored under `forcevla_all`.

```bash
python scripts/compute_norm_stats.py --config-name forcevla_lora --task flexiv_peel_cucumber_inputForce
python scripts/compute_norm_stats.py --config-name forcevla_full --task all
```

LoRA, one task or every task:

```bash
scripts/train_forcevla_lora.sh --task flexiv_peel_cucumber_inputForce
scripts/train_forcevla_lora.sh --task all
```

Full fine-tuning, one task or every task:

```bash
scripts/train_forcevla_full.sh --task flexiv_wipe_board_inputForce
scripts/train_forcevla_full.sh --task all
```

The same run through `scripts/train.py`, with the parameters the wrappers set and the ones you will usually change:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py forcevla_lora \
  --data.task=flexiv_peel_cucumber_inputForce \
  --exp-name=my_experiment \
  --overwrite \
  --batch_size=32 \
  --save_interval=2000 \
  --keep_period=10000
```

| Parameter | Meaning |
| --- | --- |
| `forcevla_lora` / `forcevla_full` | Training config name. |
| `--data.task` | One task name, or `all`. |
| `--exp-name` | Run name. Checkpoints are stored under this name. |
| `--overwrite` | Replace an existing checkpoint directory for this run. |
| `--batch_size` | Global batch size. Config default is 4. |
| `--save_interval` | Steps between checkpoints. |
| `--keep_period` | Keep checkpoints whose step is a multiple of this value. |

Use `forcevla_full` in place of `forcevla_lora` for a full fine-tune. `scripts/train_forcevla_full.sh` and `scripts/train_forcevla_lora.sh` accept `--task` only. Pass the other flags through `scripts/train.py`.

## Where things live

**Pretrained weights.** Training starts from the π₀ base checkpoint `gs://openpi-assets/checkpoints/pi0_base/params`. Both `forcevla_lora` and `forcevla_full` load it with `weight_loaders.Pi0GuidanceWeightLoader` in `src/openpi/training/config.py`.

**Checkpoints you train.** Written to `./checkpoints/<config_name>/<exp_name>/`. For example, `./checkpoints/forcevla_lora/my_experiment/`. The root is `checkpoint_base_dir` (`./checkpoints`) on `TrainConfig`.

**Normalization stats.** Written to `./assets/<config_name>/<task>/`, or `./assets/<config_name>/forcevla_all/` when `--task all`. The root is `assets_base_dir` (`./assets`) on `TrainConfig`.

**Configs and the settings that matter.**

| What | Where |
| --- | --- |
| Train configs `forcevla_lora` and `forcevla_full`: task list, batch size, step count, LoRA freeze filter, pretrained weight path | `src/openpi/training/config.py` (`FORCEVLA_TASKS`, `LeRobotForcevlaDataConfig`, `TrainConfig`) |
| Model: vision, VLM (`paligemma_variant`), action expert, action horizon | `src/openpi/models/pi0_force.py` (`Pi0_GuidanceConfig`) |
| Dataset loading, including mixed tasks | `src/openpi/training/data_loader.py` |
| Task prompt handling | `src/openpi/transforms.py` (`PromptFromLeRobotTask`) |
| Training entry point | `scripts/train.py` |
| LoRA and full wrappers | `scripts/train_forcevla_lora.sh`, `scripts/train_forcevla_full.sh` |
| Norm-stat entry point | `scripts/compute_norm_stats.py` |
