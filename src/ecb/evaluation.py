"""Post-hoc offline next-item metrics over the complete context denominator."""
import math


def ranking_metrics(rankings, targets, cutoffs=(1, 5, 10, 20)):
    rankings, targets = list(rankings), list(targets)
    if len(rankings) != len(targets) or not rankings:
        raise ValueError("rankings and targets require equal nonzero lengths")
    if any(not isinstance(k, int) or k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must be positive integers")
    result = {"count": len(targets)}
    for k in cutoffs:
        hits = ndcg = 0.
        for ranked, target in zip(rankings, targets):
            if len(set(ranked)) != len(ranked):
                raise ValueError("ranking contains duplicate item identities")
            try:
                rank = ranked.index(target) + 1
            except ValueError:
                continue
            if rank <= k:
                hits += 1
                ndcg += 1. / math.log2(rank + 1)
        result[f"hit@{k}"] = hits / len(targets)
        result[f"ndcg@{k}"] = ndcg / len(targets)
    return result
