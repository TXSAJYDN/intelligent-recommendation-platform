from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.experiments.evaluation import OfflineEvaluator


router = APIRouter(prefix="/api/admin/experiments", tags=["admin-experiments"])


@router.get("/evaluate/offline")
def offline_evaluate(
    strategy: str = Query(default="hybrid", description="推荐策略: lr/gbdt/dnn/hybrid"),
    k: int = Query(default=10, ge=1, le=50),
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    运行离线排序质量评测，计算指定策略的 AUC 和 NDCG@K。
    - AUC > 0.5 表示模型优于随机，越接近 1.0 越好
    - NDCG 衡量排序质量，1.0 为完美排序
    """
    evaluator = OfflineEvaluator(db=db, k=k, days_for_test=days)
    result = evaluator.run_evaluation(strategy=strategy)
    return result


@router.get("/evaluate/compare")
def compare_strategies(
    k: int = Query(default=10, ge=1, le=50),
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    一次性对比所有策略的排序质量
    """
    evaluator = OfflineEvaluator(db=db, k=k, days_for_test=days)
    results = {}
    for strategy in ["lr", "gbdt", "dnn", "hybrid"]:
        results[strategy] = evaluator.run_evaluation(strategy=strategy)
    return results
