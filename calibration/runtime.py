"""
推理时加载离线校准表并就地更新 ModelOutput（predictions / top_class / top_confidence）。

加载失败、单文件损坏或单条应用失败时静默跳过，行为与未启用校准时一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from models.base_model import ModelOutput


def _validate_artifact(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("artifact is not a dict")
    method = data.get("method")
    params = data.get("params") or {}
    if method == "temperature":
        if "T" not in params:
            raise ValueError("missing params.T")
    elif method == "platt":
        if "a" not in params or "b" not in params:
            raise ValueError("missing params.a/b")
    elif method == "isotonic":
        if "x_thresholds" not in params or "y_thresholds" not in params:
            raise ValueError("missing isotonic thresholds")
    else:
        raise ValueError(f"unknown method: {method}")


def load_calibration_map(
    artifacts_dir: Optional[str],
    project_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    扫描目录下 *.json，校验通过后以 artifact['model_name'] 为键装入字典。
    目录不存在、非目录或全部失败时返回 {}。
    """
    if not artifacts_dir or not str(artifacts_dir).strip():
        return {}
    path = Path(artifacts_dir)
    if project_root is not None and not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_dir():
        return {}

    # Lazy import: avoids circular import with calibration.binary_calibration
    # (binary_calibration imports models.* which pulls in model_registry -> runtime).
    from calibration.binary_calibration import apply_calibrator_dict, load_calibrator_json

    out: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(path.glob("*.json")):
        try:
            data = load_calibrator_json(fp)
            _validate_artifact(data)
            apply_calibrator_dict(0.5, data)
            name = data.get("model_name")
            if not name or not isinstance(name, str):
                continue
            out[name] = data
        except Exception:
            continue
    return out


def load_calibration_map_from_config(
    config: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Dict[str, Any]]:
    sec = config.get("calibration") or {}
    if not sec.get("enabled", False):
        return {}
    artifacts_dir = sec.get("artifacts_dir", "calibration/artifacts")
    return load_calibration_map(str(artifacts_dir), project_root=project_root)


def apply_binary_calibration_inplace(
    output: ModelOutput,
    artifact: Dict[str, Any],
) -> bool:
    """
    用单个 artifact 就地更新二分类 predictions 与 top_class/top_confidence。
    失败时返回 False 且不修改 output。
    """
    from calibration.binary_calibration import apply_calibrator_dict

    try:
        _validate_artifact(artifact)
        pos = artifact.get("positive_class_key", "1")
        neg = "0"
        probs = output.predictions
        if not probs or pos not in probs or neg not in probs:
            return False
        p_raw = float(probs[pos])
        p_cal = float(apply_calibrator_dict(p_raw, artifact))
        p_cal = max(1e-12, min(1.0 - 1e-12, p_cal))
        probs[pos] = p_cal
        probs[neg] = 1.0 - p_cal
        top = max(probs.items(), key=lambda x: x[1])
        output.top_class = top[0]
        output.top_confidence = float(top[1])
        return True
    except Exception:
        return False


def maybe_apply_calibration_map(
    output: ModelOutput,
    calibration_map: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    """
    若 calibration_map 中存在该 model_name，则尝试应用；失败或未命中则保持原 output。
    """
    if not calibration_map:
        return
    art = calibration_map.get(output.model_name)
    if not art:
        return
    apply_binary_calibration_inplace(output, art)
