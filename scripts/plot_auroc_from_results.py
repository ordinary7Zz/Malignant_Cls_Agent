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
        raise ValueError("results JSON 必须是列表格式。")
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


def _build_roc_from_samples(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        raise ValueError("未在 results JSON 中找到可用于绘制 AUROC 的样本字段。")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("真实标签只有一个类别，无法绘制 AUROC 曲线。")

    fpr, tpr, thresholds, auc_value = _compute_roc_from_scores(y_true_arr, y_prob_arr)
    return {
        "roc_curve_fpr": [float(v) for v in fpr],
        "roc_curve_tpr": [float(v) for v in tpr],
        "roc_curve_thresholds": [float(v) for v in thresholds],
        "roc_auc": float(auc_value),
        "n_aligned_samples": int(y_true_arr.shape[0]),
        "positive_class_index": 1,
    }


def _resolve_roc_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _extract_roc_summary(records)
    if summary is not None:
        fpr = summary.get("roc_curve_fpr") or []
        tpr = summary.get("roc_curve_tpr") or []
        auc_value = summary.get("roc_auc")
        if fpr and tpr and auc_value is not None:
            return summary
    return _build_roc_from_samples(records)


def _plot_roc(roc_payload: dict[str, Any], output_path: Path | None, title: str) -> None:
    fpr = np.asarray(roc_payload["roc_curve_fpr"], dtype=np.float64)
    tpr = np.asarray(roc_payload["roc_curve_tpr"], dtype=np.float64)
    auc_value = float(roc_payload["roc_auc"])
    n_samples = roc_payload.get("n_aligned_samples")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc_value:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if n_samples is None:
        plt.title(title)
    else:
        plt.title(f"{title} (n={n_samples})")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"AUROC 图已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="从 results_*.json 绘制 AUROC 曲线")
    parser.add_argument("--input", required=True, help="results_*.json 文件路径")
    parser.add_argument("--output", default=None, help="输出图片路径，可选")
    parser.add_argument("--title", default="AUROC Curve", help="图标题")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"未找到 results JSON: {input_path}")

    output_path = Path(args.output).resolve() if args.output else None
    records = _load_results(input_path)
    roc_payload = _resolve_roc_payload(records)
    _plot_roc(roc_payload, output_path, args.title)


if __name__ == "__main__":
    main()
