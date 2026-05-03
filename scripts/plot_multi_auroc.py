from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_results(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"results JSON 必须是列表格式: {path}")
    return [item for item in data if isinstance(item, dict)]


def _extract_roc_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(records):
        if item.get("record_type") == "roc_summary":
            return item
    return None


def _compute_roc_from_scores(y_true_arr: np.ndarray, y_prob_arr: np.ndarray) -> tuple[list[float], list[float], list[float], float]:
    pos_total = int(np.sum(y_true_arr == 1))
    neg_total = int(np.sum(y_true_arr == 0))
    if pos_total == 0 or neg_total == 0:
        raise ValueError("真实标签只有一个类别，无法绘制 AUROC 曲线。")

    unique_thresholds = np.unique(y_prob_arr)[::-1]
    thresholds = np.concatenate(([np.inf], unique_thresholds))

    fpr: list[float] = []
    tpr: list[float] = []
    threshold_values: list[float] = []

    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        tpr.append(tp / pos_total)
        fpr.append(fp / neg_total)
        threshold_values.append(float(threshold))

    auc_value = float(np.trapz(np.asarray(tpr, dtype=np.float64), np.asarray(fpr, dtype=np.float64)))
    return fpr, tpr, threshold_values, auc_value


def _coerce_binary_arrays(y_true: list[int], y_prob: list[float], source: Path) -> tuple[np.ndarray, np.ndarray]:
    if not y_true:
        raise ValueError(f"未在 results JSON 中找到可用于绘制 AUROC 的样本字段: {source}")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError(f"真实标签只有一个类别，无法绘制 AUROC 曲线: {source}")
    return y_true_arr, y_prob_arr


def _build_roc_from_current_schema(records: list[dict[str, Any]], source: Path) -> dict[str, Any] | None:
    y_true: list[int] = []
    y_prob: list[float] = []

    for item in records:
        if item.get("record_type", "sample") != "sample":
            continue
        true_label = item.get("true_label")
        prob_class_1 = item.get("prob_class_1")
        if true_label is None or prob_class_1 is None:
            continue
        y_true.append(int(true_label))
        y_prob.append(float(prob_class_1))

    if not y_true:
        return None

    y_true_arr, y_prob_arr = _coerce_binary_arrays(y_true, y_prob, source)
    fpr, tpr, thresholds, auc_value = _compute_roc_from_scores(y_true_arr, y_prob_arr)
    return {
        "schema": "current",
        "roc_curve_fpr": fpr,
        "roc_curve_tpr": tpr,
        "roc_curve_thresholds": thresholds,
        "roc_auc": auc_value,
        "n_samples": int(y_true_arr.shape[0]),
    }


def _build_roc_from_classification_agent_schema(records: list[dict[str, Any]], source: Path) -> dict[str, Any] | None:
    y_true: list[int] = []
    y_prob: list[float] = []

    for item in records:
        ground_truth_label = item.get("ground_truth_label")
        malignant_probability = item.get("malignant_probability")
        if ground_truth_label is None or malignant_probability is None:
            continue
        y_true.append(int(ground_truth_label))
        y_prob.append(float(malignant_probability))

    if not y_true:
        return None

    y_true_arr, y_prob_arr = _coerce_binary_arrays(y_true, y_prob, source)
    fpr, tpr, thresholds, auc_value = _compute_roc_from_scores(y_true_arr, y_prob_arr)
    return {
        "schema": "classification_agent",
        "roc_curve_fpr": fpr,
        "roc_curve_tpr": tpr,
        "roc_curve_thresholds": thresholds,
        "roc_auc": auc_value,
        "n_samples": int(y_true_arr.shape[0]),
    }


def _resolve_roc_payload(records: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    summary = _extract_roc_summary(records)
    if summary is not None:
        fpr = summary.get("roc_curve_fpr") or []
        tpr = summary.get("roc_curve_tpr") or []
        auc_value = summary.get("roc_auc")
        if fpr and tpr and auc_value is not None:
            return {
                "schema": "roc_summary",
                "roc_curve_fpr": [float(v) for v in fpr],
                "roc_curve_tpr": [float(v) for v in tpr],
                "roc_curve_thresholds": [float(v) for v in (summary.get("roc_curve_thresholds") or [])],
                "roc_auc": float(auc_value),
                "n_samples": int(summary.get("n_aligned_samples", 0)) if summary.get("n_aligned_samples") is not None else None,
            }

    payload = _build_roc_from_current_schema(records, source)
    if payload is not None:
        return payload

    payload = _build_roc_from_classification_agent_schema(records, source)
    if payload is not None:
        return payload

    raise ValueError(f"无法识别或提取可用于 AUROC 的字段: {source}")


def _default_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("results_"):
        return path.parent.name
    return stem


def _default_output_path(input_paths: list[Path]) -> Path:
    if not input_paths:
        raise ValueError("input_paths 不能为空。")
    first_input = input_paths[0]
    return first_input.parent / "multi_auroc.png"


def _plot_multi_roc(curves: list[dict[str, Any]], output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6.5))
    for curve in curves:
        fpr = np.asarray(curve["roc_curve_fpr"], dtype=np.float64)
        tpr = np.asarray(curve["roc_curve_tpr"], dtype=np.float64)
        auc_value = float(curve["roc_auc"])
        n_samples = curve.get("n_samples")
        if n_samples is None or n_samples == 0:
            label = f"{curve['label']} (AUC={auc_value:.4f})"
        else:
            label = f"{curve['label']} (AUC={auc_value:.4f}, n={n_samples})"
        plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"多曲线 AUROC 图已保存到: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="从多个 results_*.json 绘制多条 AUROC 曲线")
    parser.add_argument("--inputs", nargs="+", required=True, help="一个或多个 results_*.json 文件路径")
    parser.add_argument("--labels", nargs="*", default=None, help="每条曲线对应的显示名称，数量需与 inputs 一致")
    parser.add_argument("--output", default=None, help="输出图片路径，可选；默认保存为第一个输入文件同目录下的 multi_auroc.png")
    parser.add_argument("--title", default="Multi-AUROC Curve", help="图标题")
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"未找到 results JSON: {path}")

    if args.labels is not None and len(args.labels) > 0 and len(args.labels) != len(input_paths):
        raise ValueError("--labels 的数量必须与 --inputs 一致。")

    labels = args.labels if args.labels else [_default_label(path) for path in input_paths]
    output_path = Path(args.output).resolve() if args.output else _default_output_path(input_paths)

    curves: list[dict[str, Any]] = []
    for path, label in zip(input_paths, labels):
        records = _load_results(path)
        payload = _resolve_roc_payload(records, path)
        payload["label"] = label
        payload["source"] = str(path)
        curves.append(payload)
        print(f"已加载: {label} | schema={payload['schema']} | AUC={float(payload['roc_auc']):.4f}")

    _plot_multi_roc(curves, output_path, args.title)


if __name__ == "__main__":
    main()
