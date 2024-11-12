import re
import os

from config import AGENT_MODELS_PATH


def parse_metrics_from_model_name(model_name: str) -> dict:
    """
    Modek name example
    -------------------
    model_min_reward[1]_max_reward[2]_average_reward[3]_1710233149.7119274
    """
    model_name = model_name[5::]  # remove `model` from name
    pattern = r"(\w+)\[(-?\d+(?:\.\d+)?)\]"
    matches = re.findall(pattern, model_name)
    metrics = {}
    for match in matches:
        metric_name = match[0][1::]
        metric_value = float(match[1])
        metrics[metric_name] = metric_value
    return metrics


def get_best_model_path(dir_path: str, *, metric: str = "average_reward") -> str | None:
    if not (os.path.exists(dir_path)):
        return None
    data = []
    for model_name in os.listdir(dir_path):
        model_path = os.path.join(dir_path, model_name)
        metrics = parse_metrics_from_model_name(model_name)
        m = metrics.get(metric)
        if not (m):
            continue
        data.append((model_path, m))
    if not (data):
        return None
    best = max(data, key=lambda d: d[1])
    return best[0]


def _test():
    path = AGENT_MODELS_PATH
    name = "model_min_reward[1]_max_reward[2]_average_reward[3]_1710233149.7119274"
    # test grabber
    metrics = parse_metrics_from_model_name(name)
    print(metrics)
    # test model selector
    best_model_path = get_best_model_path(path)
    print(best_model_path)


if __name__ == "__main__":
    _test()
