# Training inputs

All paths in `configs/train.json` are relative to the directory where you run the commands. The integration reads prepared inputs; no data is distributed with this repository.

| Setting under `integration_options` | Contents |
| --- | --- |
| `minionerec_root` | The cloned MiniOneRec repository, containing `LogitProcessor.py`. |
| `base_checkpoint` | A trained MiniOneRec model and tokenizer in Hugging Face format. The tokenizer must support the catalog SIDs. |
| `ranker_checkpoint` | A frozen local Qwen causal language model and tokenizer. The ranker scores each organized candidate record using the conditional probabilities of labels 0–4. |
| `catalog` | JSONL: `item_id` (string), `sid` (complete SID string), `embedding` (numeric array), and optional `text` (string). |
| `train_contexts`, `eval_contexts` | JSONL: `request_id` (string), `history_ids` (item IDs, oldest to newest), `embedding` (request vector), and optional `text` (request text). No target fields. |
| `train_targets`, `eval_targets` | Separate JSONL files containing `request_id` and `item_id`. Identities and row order must match the corresponding context file. |
| `transitions` | JSON with `fit_split` set to `train`, and `scores` mapping each previous item ID to a dictionary of next-item IDs and nonnegative transition scores. |

Item and request vectors must use the same representation space and dimension, matching `model.embedding_dim`. Request IDs must be unique across train and evaluation. Train and evaluation contexts must be disjoint. Catalog SIDs must match the base checkpoint; collisions remain separate items with equal base-model scores and deterministic item-ID tie breaking.

`device` selects the frozen models' device (`cuda:0` by default, or `cpu`). `score_batch_size` controls MiniOneRec scoring batches, and `max_prompt_tokens` caps model input length. The small ECB controller runs on CPU. The three retrieval sources are dense similarity, lexical overlap, and train-derived item transitions.

`ranker_cost` charges each candidate scoring call; `retrieval_cost` charges each requested retrieval slot. These are explicit cost proxies in the same units as `model.budget`, not measured billing amounts. Set them to match the experiment you intend to run.

`ecb-fit-reward` generates sampled rollouts from training contexts, labels their terminal and stopped-prefix rankings with training NDCG@5, and fits the target-conditioned reward model. Evaluation targets are opened only after all evaluation trajectories finish.

This integration provides a runnable training path. Its checkpoint choices, scorer prompt, SID collision rule, and retrieval settings must be matched to the intended experiment before comparing numbers with the paper.
