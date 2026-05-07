import json
import os
import threading
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# 模型将下载并存储在项目根目录的 models 文件夹下
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _resolve_local_model_path() -> str:
    cache_root = Path(CACHE_DIR)
    model_cache_dir = cache_root / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    ref_file = model_cache_dir / "refs" / "main"
    if ref_file.exists():
        snapshot_id = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = model_cache_dir / "snapshots" / snapshot_id
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshots_dir = model_cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(snapshots[0])

    return MODEL_NAME


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_resolve_local_model_path(), cache_folder=CACHE_DIR)
    return _model


def build_item_text(title: str | None, category: str | None, tags: str | None, description: str | None) -> str:
    parts = [title or "", category or "", tags or "", description or ""]
    return " ".join([x.strip() for x in parts if x and x.strip()])


def encode_text(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros((384,), dtype=np.float32)
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def serialize_vector(vector: np.ndarray) -> str:
    return json.dumps(vector.astype(np.float32).tolist(), ensure_ascii=False)


def deserialize_vector(raw: str) -> np.ndarray:
    arr = np.asarray(json.loads(raw), dtype=np.float32)
    if arr.ndim != 1:
        return arr.flatten().astype(np.float32)
    return arr
