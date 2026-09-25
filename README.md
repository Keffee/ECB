# ECB

ECB (Evidence-Conditioned Continuous Bandit) learns when to expand a recommendation shortlist and how to divide the retrieval budget. Our experiments use MiniOneRec as the frozen base recommender.

## Install

Use Python 3.10 or later. Clone ECB and MiniOneRec into the same directory:

```bash
git clone https://github.com/Keffee/ECB.git
git clone https://github.com/AkaliKong/MiniOneRec.git
git -C MiniOneRec checkout 8e03e354033fc81f830580f01c102bd7fbaa262a
cd ECB
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[onerec]'
```

## Run

In `configs/train.json`, set `base_checkpoint` to your trained MiniOneRec checkpoint and `ranker_checkpoint` to your frozen local Qwen ranker checkpoint. Both directories must include the model and tokenizer. Set the catalog, train/evaluation context, target, and transition file paths under `integration_options`, and set `model.embedding_dim` to the size of your item embeddings. [Input fields](docs/inputs.md) lists the expected file contents. Models and data are not included.

Run these commands from the ECB directory. Select an available GPU before starting; the default config uses `cuda:0` within the selected GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0
ecb-fit-reward --config configs/train.json --output outputs/reward
ecb-train --config configs/train.json
ecb-evaluate --config configs/train.json \
  --checkpoint outputs/controller/controller.pt \
  --output outputs/evaluation
```

The first command samples training trajectories and fits the reward model. The second trains the ECB controller while keeping the recommender and ranker frozen. The last evaluates the saved controller.

The controller is saved to `outputs/controller/controller.pt`; evaluation metrics are in `outputs/evaluation/metrics.json`. For another run, choose new output directories and update `reward_checkpoint` and `output_dir` in the config to match.
