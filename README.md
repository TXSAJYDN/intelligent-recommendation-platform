# 智能内容推荐平台

一套完整的企业级推荐系统，覆盖数据采集、多路召回、多模型排序、离线评估、A/B 实验与实时推荐全链路。基于 FastAPI 构建，开箱即用。

---

## 系统架构

```
用户行为 → 画像更新 → 多路召回 → 精排模型 → 缓存 → 推荐结果
              ↑                                    ↓
         行为写入触发缓存失效 ←──────────────────────┘
```

### 数据层

- 用户、内容、行为日志持久化存储（SQLite / 可切换 PostgreSQL）
- 用户画像自动聚合（偏好分类、偏好标签、活跃度、最近活跃时间）
- Embedding 向量化存储与 Faiss 索引

### 召回层

- **UserCF**：基于用户相似度协同过滤
- **ItemCF**：基于物品共现协同过滤
- **Embedding 召回**：基于 SentenceTransformer 语义向量相似度
- **Faiss 召回**：基于向量索引的高效近邻检索

### 排序层

- **LR（逻辑回归）**：线性特征加权打分
- **GBDT**：非线性交叉特征打分（兴趣×热度、新鲜度×兴趣、热度衰减）
- **DNN**：PyTorch 深度排序模型，支持训练、标准化、版本管理
- **Hybrid**：多模型融合策略（`0.2*LR + 0.2*GBDT + 0.6*DNN`）

### 特征工程

- `recall_score`：召回阶段得分
- `popularity_norm`：归一化热度
- `freshness`：时效性衰减
- `match_interest`：用户兴趣与物品标签匹配
- `behavior_weight`：用户对该物品的历史行为强度
- `user_behavior_count`：用户总行为数
- `user_recent_7d_count`：用户近 7 日行为数
- `item_ctr_like`：物品 like 转化率
- `item_ctr_collect`：物品 collect 转化率
- `item_age_days_norm`：物品年龄归一化

### 实验层

- **A/B/C 分组实验**：用户按组绑定不同排序策略
  - A 组 → LR
  - B 组 → GBDT
  - C 组 → DNN
- **离线评测**：基于 `AUC + NDCG@K` 衡量排序质量
- **业务指标**：按组统计 CTR、Like Rate、Collect Rate、Engagement Rate

### 缓存与实时生效

- Redis / 内存双模式缓存（自动降级）
- 用户行为写入后精准失效该用户推荐缓存
- 分场景 TTL 策略：
  - 实时推荐：30s
  - 常规推荐：120s
- 保证实时性的同时维护系统吞吐

---

## 离线评测结果

各排序策略在 engagement-level comparison 评测下的表现（正样本 = 深度互动，负样本 = 浅度浏览）：

| 策略 | avg_auc | avg_ndcg |
|------|---------|----------|
| LR | 0.7319 | 0.9091 |
| GBDT | 0.7367 | 0.9348 |
| DNN | 0.8011 | 0.9417 |

结论：DNN 排序质量显著优于 LR/GBDT，适合作为线上主策略。

---

## 目录结构

```text
intelligent-recommendation-platform/
├── backend/
│   ├── cache/                 # Redis/内存缓存适配层
│   ├── data/                  # CSV 种子数据（用户、内容、行为）
│   ├── experiments/           # 离线评估与 A/B 实验
│   ├── ranking/               # LR/GBDT/DNN 排序模型与特征工程
│   ├── recall/                # UserCF/ItemCF/Embedding/Faiss 召回
│   ├── routers/               # FastAPI 路由层
│   ├── services/              # 推荐服务、画像服务、热门服务
│   ├── main.py                # 应用入口与启动逻辑
│   ├── models.py              # ORM 数据模型
│   ├── crud.py                # 数据库操作
│   ├── schemas.py             # Pydantic 数据结构
│   └── database.py            # 数据库连接配置
├── frontend/                  # 前端页面（HTML/CSS/JS）
├── models/
│   └── ranking/dnn/           # DNN 模型权重与元数据
├── requirements.txt
└── README.md
```

---

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务（首次启动自动建表、灌入种子数据、加载模型）
uvicorn backend.main:app --reload

# 3. 访问接口文档
# http://127.0.0.1:8000/docs
```

---

## 核心接口

### 推荐

| 接口 | 说明 |
|------|------|
| `GET /api/recommend/guess/{user_id}` | 猜你喜欢（A/B 分组策略） |
| `GET /api/recommend/realtime/{user_id}` | 实时推荐（短缓存，行为后秒级刷新） |
| `GET /api/recommend/ranking/{user_id}?model=dnn` | 指定排序模型推荐 |
| `GET /api/recommend/faiss/{user_id}` | Faiss 向量召回 + DNN 排序 |
| `GET /api/recommend/hot` | 热门推荐 |

### 行为

| 接口 | 说明 |
|------|------|
| `POST /api/behaviors` | 记录用户行为 |
| `POST /api/behaviors/realtime` | 实时行为上报（触发缓存失效） |
| `GET /api/behaviors/{user_id}` | 查询用户行为历史 |

### 评测与实验

| 接口 | 说明 |
|------|------|
| `GET /api/admin/experiments/evaluate/offline?strategy=dnn&k=10` | 单策略离线评测 |
| `GET /api/admin/experiments/evaluate/compare` | 全策略对比评测 |
| `GET /api/stats/abtest` | A/B 分组流量分布 |
| `GET /api/stats/abtest/metrics` | 按组 CTR/Like/Collect 转化率 |

### 管理

| 接口 | 说明 |
|------|------|
| `GET /api/admin/ranking/status` | DNN 模型状态 |
| `POST /api/admin/ranking/train` | 触发 DNN 模型训练 |
| `POST /api/admin/ranking/reload` | 热加载最新模型 |

---

## 实时推荐验证

```
1. GET  /api/recommend/realtime/1        → 记录推荐结果 A
2. POST /api/behaviors/realtime          → 上报用户 1 对某物品的 like 行为
3. GET  /api/recommend/realtime/1        → 记录推荐结果 B
```

预期：结果 B 与结果 A 不同（排序或候选变化），验证"行为写入 → 缓存失效 → 重新计算"链路已生效。

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Web 框架 | FastAPI |
| ORM | SQLAlchemy |
| 数据库 | SQLite（默认）/ PostgreSQL（可配） |
| 缓存 | Redis（可选）/ 进程内存（自动降级） |
| 向量模型 | SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) |
| 向量索引 | Faiss |
| 深度排序 | PyTorch |
| 前端 | 原生 HTML/CSS/JS |

---

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DATABASE_URL` | `sqlite:///./app.db` | 数据库连接 |
| `REDIS_URL` | 空（使用内存缓存） | Redis 连接地址 |

---

## 后续演进方向

- 接入曝光日志，建立真实 CTR/CVR 指标口径
- 定时离线训练 + 模型自动回滚机制
- 特征仓与特征版本管理，保证线上线下一致性
- 引入多目标排序（点击 + 时长 + 收藏联合优化）
- 前端可视化大盘（实验效果、模型指标、系统健康度）
