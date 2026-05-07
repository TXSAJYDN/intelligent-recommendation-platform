def assign_group(user_id: int) -> str:
    rem = user_id % 3
    if rem == 0:
        return "A"
    elif rem == 1:
        return "B"
    else:
        return "C"


def strategy_by_group(group_name: str) -> str:
    """
    实验策略映射：
    - A组 (对照组): 线性回归排序 (LR)
    - B组 (进阶组): 树模型排序 (GBDT)
    - C组 (核心组): 深度学习排序 (DNN)
    """
    mapping = {
        "A": "lr",
        "B": "gbdt",
        "C": "dnn"
    }
    return mapping.get(group_name, "hybrid")
