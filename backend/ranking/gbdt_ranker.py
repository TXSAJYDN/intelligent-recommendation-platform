def score_with_gbdt(features: dict[str, float], recall_score: float) -> float:
    match = features.get("match_interest", 0.0)
    pop_norm = features.get("popularity_norm", min(features.get("popularity", 0.0) / 1000.0, 1.0))
    freshness = features.get("freshness", 0.0)
    bw = features.get("behavior_weight", 0.0)
    age_norm = features.get("item_age_days_norm", 0.5)

    # 非线性交叉特征 —— 与 lr 形成差异化排序
    cross_match_pop = match * pop_norm          # 兴趣命中 × 热度
    cross_fresh_match = freshness * match       # 新鲜度 × 兴趣
    pop_decay = pop_norm * (1.0 - age_norm)     # 热度随时间衰减

    base = 0.25 * recall_score + 0.15 * match + 0.1 * pop_norm + 0.1 * freshness
    interaction = 0.15 * cross_match_pop + 0.1 * cross_fresh_match + 0.1 * pop_decay
    bonus = 0.05 * min(bw, 1.0)
    return max(0.0, min(1.0, base + interaction + bonus))
