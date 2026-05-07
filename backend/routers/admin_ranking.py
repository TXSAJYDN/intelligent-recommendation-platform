from fastapi import APIRouter, HTTPException, Query

from backend.ranking.dnn_ranker import get_dnn_status, load_dnn_model
from backend.ranking.train_dnn import train


router = APIRouter(prefix="/api/admin/ranking", tags=["admin-ranking"])


@router.get("/status")
def ranking_status():
    return get_dnn_status()


@router.post("/reload-dnn")
def reload_dnn():
    ok = load_dnn_model(force_reload=True)
    return {
        "ok": ok,
        "status": get_dnn_status(),
    }


@router.post("/train-dnn")
def train_dnn(
    epochs: int = Query(default=20, ge=1, le=200),
    batch_size: int = Query(default=256, ge=16, le=4096),
    learning_rate: float = Query(default=1e-3, gt=0.0, le=1.0),
    neg_ratio: int = Query(default=3, ge=1, le=20),
    seed: int = Query(default=42, ge=1, le=999999),
):
    try:
        result = train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            neg_ratio=neg_ratio,
            seed=seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    load_dnn_model(force_reload=True)
    return {
        "ok": True,
        "train_result": result,
        "status": get_dnn_status(),
    }
