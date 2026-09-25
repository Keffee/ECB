# ECB

ECB (Evidence-Conditioned Continuous Bandit) controls when a recommender should spend more effort finding candidates.

At each round, the base recommender produces a ranked list. A gate either returns that list or requests candidate scores from an Evidence-aware Ranker. On an upgrade, a joint policy chooses how to extract interests, mix evidence, and divide the retrieval budget. New candidates are added to the pool, and the base recommender ranks them again.

This repository contains the controller, training code, and evaluation loop. Bring your own prepared inputs, base recommender, candidate scorer, and retrieval sources. If you use OpenOneRec, obtain and set it up separately.

## Install

Use Python 3.10 or later and install a PyTorch build suitable for your environment.

```bash
pip install -e ".[test]"
pytest -q
```

## Connect your components

Set `integration` in `configs/train.json` to a Python factory in your project:

```json
"integration": "my_project.ecb_adapter:build"
```

The factory is called as `build(options, config)`. It returns an object with these components:

| Component | What it supplies |
|---|---|
| `base.rank(state, pool, limit)` | Candidates in base-recommender order, with aligned logits |
| `ranker.score(state, candidates)` | One score per exposed candidate, in the same order |
| `retriever.retrieve(state, signal, quotas, excluded)` | Candidates from three retrieval sources |
| `training_records()` | Prepared training contexts and their target embeddings |
| `reward_examples()` | Sampled training trajectories with target embeddings and utility labels |
| `evaluation_contexts()` | Evaluation contexts without targets |
| `evaluation_targets()` | Targets used only after inference has finished |

The record types and adapter signatures are in `src/ecb/types.py` and `src/ecb/interfaces.py`. Keep the base recommender, representations, ranker, and retrieval components frozen during controller training.

For a pointwise scoring service, `OrganizedEvidenceRanker(provider)` assembles each candidate's observed history, request text, item text, and retrieval evidence. The provider implements `estimate_cost(record)` and `score(record)`; see `src/ecb/ranker.py`.

Both the ranker and retriever provide `estimate_cost` before execution and return actual usage afterward. Choose one budget unit for both components, such as a request-based or token-based cost proxy. Their estimates must cover the maximum allowed usage, including retries. Raise `ServiceFailure` with the incurred cost if a service fails.

## Train

Set the embedding size, budgets, integration factory, and output paths in `configs/train.json`.

First fit the trajectory reward model using training records:

```bash
ecb-fit-reward --config configs/train.json --output outputs/reward
```

Then train the controller:

```bash
ecb-train --config configs/train.json
```

The reward model is frozen before controller training. The policy trains on sampled trajectories using reward-to-go REINFORCE and a separate value baseline. Gate fitting runs separately and uses the net value of an executed continuation compared with stopping. The first training epoch uses forced upgrades to warm up the policy.

Training writes `controller.pt`, epoch summaries, per-round traces, and a manifest to `output_dir`. Use a new output directory for each run.

## Evaluate

Use the same model configuration as the saved checkpoint:

```bash
ecb-evaluate --config configs/train.json \
  --checkpoint outputs/controller/controller.pt \
  --output outputs/evaluation
```

Evaluation samples actions with the configured seed. Add `--deterministic` to use conditional distribution means. The metrics include all evaluation contexts, including failed and empty recommendations. Targets are read only after the recommendation workflows finish.

## Core modules

- `workflow.py`: gate-first execution, candidate pools, stopping, and cost accounting.
- `ranker.py`: organized candidate evidence for a pointwise scoring provider.
- `policy.py`: masked candidate evidence and the conditional three-head policy.
- `retrieval.py`: interest extraction, evidence mixing, and source quotas.
- `reward.py`: target-conditioned trajectory reward.
- `training.py`: policy, critic, and gate updates.
- `evaluation.py`: offline Hit@k and NDCG@k.

The method mapping and concrete action parameterization are in [docs/method.md](docs/method.md).

## License

MIT.
