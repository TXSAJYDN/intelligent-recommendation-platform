import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import models
from backend.ranking.dnn_ranker import score_with_dnn
from backend.ranking.feature_engineering import build_features
from backend.ranking.gbdt_ranker import score_with_gbdt
from backend.ranking.lr_ranker import score_with_lr


def _score_item(db: Session, user_id: int, item_id: int, strategy: str) -> float:
    """用指定策略为 user-item 对打分"""
    features = build_features(db=db, user_id=user_id, item_id=item_id, recall_score=0.5)
    leakage_keys = {
        "behavior_weight",
        "user_behavior_count",
        "user_recent_7d_count",
        "item_ctr_like",
        "item_ctr_collect",
    }
    for key in leakage_keys:
        features[key] = 0.0
    if strategy == "lr":
        return score_with_lr(features, 0.5)
    elif strategy == "gbdt":
        return score_with_gbdt(features, 0.5)
    elif strategy == "dnn":
        return score_with_dnn(features, 0.5)
    else:
        lr_s = score_with_lr(features, 0.5)
        gbdt_s = score_with_gbdt(features, 0.5)
        dnn_s = score_with_dnn(features, 0.5)
        return 0.2 * lr_s + 0.2 * gbdt_s + 0.6 * dnn_s


def _compute_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """计算 AUC：正样本得分高于负样本的概率"""
    if not pos_scores or not neg_scores:
        return 0.5
    correct = 0
    total = len(pos_scores) * len(neg_scores)
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                correct += 1
            elif ps == ns:
                correct += 0.5
    return correct / total if total > 0 else 0.5


def _compute_ndcg(scores_with_labels: list[tuple[float, int]], k: int) -> float:
    """计算 NDCG@K"""
    # 按模型打分降序排列
    sorted_items = sorted(scores_with_labels, key=lambda x: x[0], reverse=True)[:k]

    # DCG
    dcg = 0.0
    for i, (_, label) in enumerate(sorted_items):
        dcg += label / np.log2(i + 2)

    # Ideal DCG
    ideal_sorted = sorted(scores_with_labels, key=lambda x: x[1], reverse=True)[:k]
    idcg = 0.0
    for i, (_, label) in enumerate(ideal_sorted):
        idcg += label / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


class OfflineEvaluator:
    def __init__(self, db: Session, k: int = 10, days_for_test: int = 7, neg_ratio: int = 3):
        self.db = db
        self.k = k
        self.days_for_test = days_for_test
        self.neg_ratio = neg_ratio

    def run_evaluation(self, strategy: str = "hybrid") -> dict:
        """
        排序质量评估（engagement-level comparison）：
        - 深度互动 Item (like/collect, weight>=3) 为正样本
        - 浅度互动 Item (view only, weight=1) 为负样本
        - 测试模型能否给深度互动 Item 打更高分
        - 这避免了 behavior_weight 特征的标签泄露问题
        """
        # 1. 获取所有行为
        behaviors = list(
            self.db.scalars(
                select(models.UserBehavior).order_by(models.UserBehavior.created_at.asc())
            ).all()
        )
        if not behaviors:
            return {"error": "No behavior data found"}

        # 2. 构建用户-Item 最大权重映射
        user_item_max_weight: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for b in behaviors:
            w = float(b.weight or 1.0)
            if w > user_item_max_weight[b.user_id][b.item_id]:
                user_item_max_weight[b.user_id][b.item_id] = w

        # 3. 对每个用户划分正样本（深度互动）和负样本（浅度互动）
        all_aucs = []
        all_ndcgs = []
        total_users = 0
        total_pos = 0
        total_neg = 0

        for user_id, item_weights in user_item_max_weight.items():
            # 正样本：最大权重 >= 3（like/collect）
            pos_items = [iid for iid, w in item_weights.items() if w >= 3.0]
            # 负样本：最大权重 == 1（只有 view）
            neg_items = [iid for iid, w in item_weights.items() if w <= 1.0]

            if not pos_items or not neg_items:
                continue

            # 打分
            pos_scores = [_score_item(self.db, user_id, iid, strategy) for iid in pos_items]
            neg_scores = [_score_item(self.db, user_id, iid, strategy) for iid in neg_items]

            # AUC
            auc = _compute_auc(pos_scores, neg_scores)
            all_aucs.append(auc)

            # NDCG@K
            scores_labels = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
            ndcg = _compute_ndcg(scores_labels, self.k)
            all_ndcgs.append(ndcg)

            total_users += 1
            total_pos += len(pos_items)
            total_neg += len(neg_items)

        if total_users == 0:
            return {"error": "No users with both deep and shallow interactions found"}

        # 4. 汇总
        return {
            "strategy": strategy,
            "k": self.k,
            "test_users": total_users,
            "total_positive_pairs": total_pos,
            "total_negative_pairs": total_neg,
            "avg_auc": round(float(np.mean(all_aucs)), 4),
            "avg_ndcg": round(float(np.mean(all_ndcgs)), 4),
            "max_auc": round(float(np.max(all_aucs)), 4),
            "min_auc": round(float(np.min(all_aucs)), 4),
            "interpretation": "AUC>0.5 means model ranks deeply-engaged items above view-only items",
            "timestamp": datetime.utcnow().isoformat()
        }
