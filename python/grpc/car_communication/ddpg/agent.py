from collections import deque
import numpy as np
import random
import time
import re
import os

from config import AGENT_LOGS_PATH, AGENT_MODELS_PATH
from .parameters import *

#model_min_reward[-1.54]_max_reward[28.44]_average_reward[13.347]_1710233149.7119274.keras
def parse_metrics_from_model_name(model_name:str) -> dict:
    model_name = model_name[5::] # remove `model` from name
    pattern = r"(\w+)\[(-?\d+\.\d+)\]"
    matches = re.findall(pattern, model_name)
    metrics = {}
    for match in matches:
        metric_name = match[0][1::]
        metric_value = float(match[1])
        metrics[metric_name] = metric_value
    return metrics

def get_best_model_path(*, models_path: str | None = None, metric:str = 'average_reward') -> str:
    models_path =  models_path if models_path else AGENT_MODELS_PATH
    data = []
    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        metrics = parse_metrics_from_model_name(model_name)
        m = metrics.get(metric)
        if not(m): continue 
        data.append((model_path, m))
    if not(data): return ''
    best = max(data, key=lambda d: d[1])
    return best[0]

def extract_inputs(data: list) -> list[np.ndarray]:
    return [np.array(column) for column in zip(*data)]

class DDPGAgent:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

def _test():
    pass

if __name__ == '__main__':
    _test()
