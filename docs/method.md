# Method and implementation

| Paper component | Implementation |
|---|---|
| Persistent history and request context, latest retrieval signal | `Context` and `State`; history is capped, signal is replaced |
| Gate before Evidence-aware Ranker | `Workflow.run`; STOP returns the current base ranking |
| Organized pointwise scorer input | `OrganizedEvidenceRanker`; observed history, context, candidate text and construction evidence |
| Candidate-indexed score distribution | `masked_distribution`; softmax over valid candidates only |
| Joint continuous action | `JointPolicy`; conditional λ, then ρ given λ, then ω given λ and ρ |
| Candidate expansion and base reranking | `Workflow.run`; bounded retained pool, no final ranker pass |
| Frozen train-only target-conditioned feedback | `TrajectoryReward`; actual terminal list and signal are inputs |
| Likelihood-ratio training | `Trainer.actor_batch`; sampled actions, reward-to-go, detached baseline |
| Separate gate training | `Trainer.gate_batch`; executed continuation net utility minus stop utility |
| Held-out target isolation | Separate inference and training types; evaluation reads targets afterward |

## Declared parameterization

The paper specifies the roles of the three actions. This implementation uses:

- **λ:** a Beta sample in (0, 1). History embeddings are pooled with recency weights proportional to `exp(10 * λ * position)`, where positions range from -1 to 0.
- **ρ:** a three-dimensional Dirichlet sample. It mixes normalized history interest, request context, and score-weighted candidate evidence.
- **ω:** a three-dimensional Dirichlet sample. It divides the current retrieval slots among three externally supplied sources using largest-remainder rounding.

These are concrete implementation choices; the paper does not uniquely prescribe these distribution families or the recency formula. Source order belongs to the integration configuration. Zero quotas are allowed. Every upgrade samples all three heads, and the joint log density is the sum of their conditional log densities. Samples are detached from the density parameters; gradients do not pass through retrieval or ranking.

The pre-score gate features contain history, context, latest signal, base-weighted candidates, retained-pool summary, remaining budget, remaining rounds, candidate counts, base-score entropy and margin. After scoring, the remaining budget is updated before it is passed to the actor and critic. The actor additionally pools candidate embeddings paired with the freshly obtained relative score distribution. Padded candidates contribute nothing.

## Training schedule

1. Fit the reward model on scored, sampled training records and freeze it.
2. Sample fresh rollouts for each actor update. Warmup uses forced upgrades; subsequent epochs use the fixed current gate.
3. Train the critic on Monte Carlo reward-to-go. Actor advantages use the detached, action-independent baseline. Per-step actor terms are summed within each trajectory and averaged over all sampled trajectories.
4. With the actor fixed, execute train-only continuation probes and fit the gate. For each reached state, the label is terminal utility minus all subsequent realized costs minus stop utility. This is a sampled estimate of the stopping criterion, not a claim of an optimal gate.
5. Freeze the resulting controller for evaluation.

The entropy regularizer includes the score-function contribution of upstream sampled heads to later conditional entropies. The reward model uses the actual terminal ranking and bounded final state; candidate ranker scores are never used as training rewards. Separate train-stop estimates are used only for gate labels, not to expose a full counterfactual action table to the actor.

## Budgets and failures

Ranker and retrieval cost bounds are checked before each corresponding call. If scoring leaves insufficient retrieval budget, the workflow returns the last completed base ranking and retains the scoring cost. Reported bound violations also stop the workflow and retain the actual cost. Adapters are responsible for enforcing their own retry caps.

Each STOP, upgrade, and execution-limit decision has a trace entry with base scores, gate value, threshold and remaining budget. Steps retain separate ranker/retrieval costs and source counts. Usage separates attempted logical user-item evaluations, physical requests, retries, cache hits, tokens, latency, and billed amounts. The paper's nominal exposed-pair count is the sum of the candidate-list lengths on upgrade steps; actual attempted logical evaluations may be smaller if a scoring pass fails partway through. Both are recoverable from the trace. The budget scalar `units` must have one consistent, declared meaning. Base-recommender inference is outside the paper's ranker-plus-retrieval cost objective and is not silently treated as either service.

After a successful retrieval, even on the last allowed round, the base recommender reranks the pool. If retrieval or reranking fails, the last complete ranking is returned. Evaluation preserves every context in the metric denominator.

## Scope

This release implements the core protocol and includes synthetic correctness tests. It does not bundle datasets, input preparation, pretrained recommenders, checkpoints, service credentials, or historical experiment results. Adapters must provide observed, target-free inference inputs; typed interfaces cannot detect future information hidden inside external embeddings or text.
