"""
二分类（class_0 / class_1）概率校准：离线拟合与在线应用。

支持方法（均仅依赖原始 class_1 概率 p∈(0,1)）：
- temperature: 对 logit(p) 做温度缩放  p_cal = σ(logit(p)/T)
- platt: Platt 缩放  p_cal = σ(a·logit(p)+b)，与单特征逻辑回归等价
- isotonic: 保序回归（sklearn IsotonicRegression）

校准器以 JSON 落盘，供推理阶段加载。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from typing_extensions import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from models.base_model import ModelOutput

MethodName = Literal["temperature", "platt", "isotonic"]


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def extract_p_malignant(
    pred: ModelOutput,
    positive_key: str = "1",
) -> Optional[float]:
    """
    从 ModelOutput 中取 P(class_1)。若缺少对应键则返回 None。
    """
    probs = pred.predictions or {}
    if positive_key not in probs:
        return None
    return float(probs[positive_key])


def binary_nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """平均负对数似然（class_1 为正类）。"""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """期望校准误差（ECE），分箱为等宽于 [0,1]。"""
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        w = float(np.mean(m))
        ece += w * abs(acc - conf)
    return ece


def apply_temperature(p_raw: Union[float, np.ndarray], T: float) -> np.ndarray:
    T = float(T)
    if T <= 0:
        raise ValueError("Temperature T must be positive")
    p_raw = np.asarray(p_raw, dtype=np.float64)
    z = _logit(p_raw)
    return _sigmoid(z / T)


def apply_platt(p_raw: Union[float, np.ndarray], a: float, b: float) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=np.float64)
    z = _logit(p_raw)
    return _sigmoid(a * z + b)


def apply_isotonic(
    p_raw: Union[float, np.ndarray],
    x_thresholds: List[float],
    y_thresholds: List[float],
) -> np.ndarray:
    """使用训练时保存的保序回归阈值做预测（与 sklearn out_of_bounds='clip' 一致）。"""
    x = np.asarray(x_thresholds, dtype=np.float64).ravel()
    y = np.asarray(y_thresholds, dtype=np.float64).ravel()
    p_raw = np.asarray(p_raw, dtype=np.float64)
    shape = p_raw.shape
    flat = p_raw.reshape(-1)
    xmin, xmax = float(x.min()), float(x.max())
    flat = np.clip(flat, xmin, xmax)
    out = np.interp(flat, x, y)
    return out.reshape(shape)


def fit_temperature_scaling(p_raw: np.ndarray, y: np.ndarray) -> float:
    """
    在验证集上拟合温度 T>0，使 NLL 最小：p_cal = σ(logit(p)/T)。
    """
    p_raw = np.asarray(p_raw, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(p_raw) != len(y):
        raise ValueError("p_raw and y length mismatch")

    def nll(T: float) -> float:
        if T <= 1e-8:
            return 1e12
        p_hat = apply_temperature(p_raw, T)
        return binary_nll(y, p_hat)

    res = minimize_scalar(nll, bounds=(1e-3, 100.0), method="bounded")
    T_opt = float(res.x)
    if not np.isfinite(T_opt) or T_opt <= 0:
        return 1.0
    return T_opt


def fit_platt_scaling(p_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """单特征 Platt：p_cal = σ(a·logit(p)+b)。"""
    p_raw = np.asarray(p_raw, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = _logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return a, b


def fit_isotonic_regression(
    p_raw: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ir = IsotonicRegression(out_of_bounds="clip")
    p_raw = np.asarray(p_raw, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ir.fit(p_raw, y)
    return ir.X_thresholds_, ir.y_thresholds_


@dataclass
class FitResult:
    method: MethodName
    params: Dict[str, Any]
    metrics_raw: Dict[str, float]
    metrics_cal: Dict[str, float]
    n_samples: int


def fit_binary_calibrator(
    p_raw: np.ndarray,
    y: np.ndarray,
    method: MethodName = "platt",
) -> FitResult:
    """
    在验证集上拟合二分类校准器并报告 raw / calibrated 的 NLL、Brier、ECE。
    """
    p_raw = np.asarray(p_raw, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(p_raw) != len(y):
        raise ValueError("p_raw and y length mismatch")
    if len(y) < 2:
        raise ValueError("Need at least 2 samples to fit a calibrator")
    if np.unique(y).size < 2:
        raise ValueError("Need both classes in y to fit a calibrator")

    m_raw = {
        "nll": binary_nll(y, p_raw),
        "brier": binary_brier(y, p_raw),
        "ece": expected_calibration_error(y, p_raw),
    }

    if method == "temperature":
        T = fit_temperature_scaling(p_raw, y)
        p_cal = apply_temperature(p_raw, T)
        params = {"T": T}
    elif method == "platt":
        a, b = fit_platt_scaling(p_raw, y)
        p_cal = apply_platt(p_raw, a, b)
        params = {"a": a, "b": b}
    elif method == "isotonic":
        xt, yt = fit_isotonic_regression(p_raw, y)
        p_cal = apply_isotonic(p_raw, xt.tolist(), yt.tolist())
        params = {
            "x_thresholds": [float(x) for x in xt],
            "y_thresholds": [float(x) for x in yt],
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    m_cal = {
        "nll": binary_nll(y, p_cal),
        "brier": binary_brier(y, p_cal),
        "ece": expected_calibration_error(y, p_cal),
    }

    return FitResult(
        method=method,
        params=params,
        metrics_raw=m_raw,
        metrics_cal=m_cal,
        n_samples=len(y),
    )


def apply_calibrator_dict(p_raw: float, cal: Dict[str, Any]) -> float:
    """
    根据 save_calibrator_json 保存的字典对单个 p_raw 做校准。
    """
    method = cal.get("method")
    params = cal.get("params") or {}
    if method == "temperature":
        return float(apply_temperature(p_raw, float(params["T"])))
    if method == "platt":
        return float(apply_platt(p_raw, float(params["a"]), float(params["b"])))
    if method == "isotonic":
        return float(
            apply_isotonic(
                p_raw,
                params["x_thresholds"],
                params["y_thresholds"],
            )
        )
    raise ValueError(f"Unknown calibrator method: {method}")


def build_calibrator_artifact(
    model_name: str,
    fit: FitResult,
    positive_class_key: str = "1",
    schema_version: int = 1,
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "model_name": model_name,
        "task": "binary_thyroid_malignancy",
        "positive_class_key": positive_class_key,
        "method": fit.method,
        "params": fit.params,
        "n_samples": fit.n_samples,
        "metrics_raw": fit.metrics_raw,
        "metrics_cal": fit.metrics_cal,
    }


def save_calibrator_json(artifact: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)


def load_calibrator_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
