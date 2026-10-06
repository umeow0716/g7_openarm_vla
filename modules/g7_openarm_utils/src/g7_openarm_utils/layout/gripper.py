from __future__ import annotations

import numpy as np

def motor_to_openness(q: float, close_q: float, open_q: float):
    openness = (q - close_q) / (open_q - close_q)
    return np.clip(openness, 0.0, 1.0)

def openness_to_motor(openness: float, close_q: float, open_q: float):
    openness = np.clip(openness, 0.0, 1.0)
    return close_q + openness * (open_q - close_q)

def qpos_to_openness(q: float):
    return np.clip(q / 0.045, 0.0, 1.0)

def openness_to_qpos(openness: float):
    openness = np.clip(openness, 0.0, 1.0)
    return openness * 0.045

def openness_rate_to_qvel(dopenness: float):
    return dopenness * 0.045

def qvel_to_openness_rate(qvel: float):
    return qvel / 0.045
