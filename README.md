# ECB

ECB (Evidence-Conditioned Continuous Bandit) learns when to expand recommendation candidates and how to allocate the retrieval budget.

## Install

Use Python 3.10 or later. From the repository directory, run:

```bash
pip install -e .
```

## Start training

Use your own prepared inputs and recommendation components. If you use OpenOneRec, set it up separately.

In `configs/train.json`, set `integration` to your component adapter and update the model settings and output paths. The adapter interface is defined in [interfaces.py](src/ecb/interfaces.py).

Fit the reward model, then train the controller:

```bash
ecb-fit-reward --config configs/train.json --output outputs/reward
ecb-train --config configs/train.json
```

The controller is saved to `outputs/controller/controller.pt`. Use new output directories for subsequent runs.
