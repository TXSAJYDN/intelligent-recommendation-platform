def score_with_lr(features: dict[str, float], recall_score: float) -> float:
    return (
        0.35 * recall_score
        + 0.25 * features.get("match_interest", 0.0)
        + 0.2 * min(features.get("popularity", 0.0) / 100.0, 1.0)
        + 0.1 * features.get("freshness", 0.0)
        + 0.1 * features.get("behavior_weight", 0.0)
    )
