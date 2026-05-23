from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_results(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
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


def _load_doctor_points(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"医生性能 JSON 必须是对象格式: {path}")
    if data.get("record_type") != "doctor_multi_task_roc_point_summary":
        raise ValueError(f"无法识别的医生性能 JSON: {path}")

    plot_points = data.get("plot_points")
    if not isinstance(plot_points, list):
        raise ValueError(f"医生性能 JSON 缺少 plot_points 列表: {path}")

    points: list[dict[str, Any]] = []
    for item in plot_points:
        if not isinstance(item, dict):
            continue
        roc_point = item.get("roc_point") or {}
        fpr = roc_point.get("fpr")
        tpr = roc_point.get("tpr")
        if fpr is None or tpr is None:
            continue

        task_label = str(item.get("task_label") or "")
        doctor_label = str(item.get("doctor_label") or "")
        points.append(
            {
                "task_label": task_label,
                "doctor_label": doctor_label,
                "label": str(item.get("label") or f"{task_label} - {doctor_label}"),
                "fpr": float(fpr),
                "tpr": float(tpr),
                "n_samples": item.get("n_samples"),
                "metrics": item.get("metrics") or {},
                "source": str(path),
            }
        )

    return points


def _filter_doctor_points_for_task(
    doctor_points: list[dict[str, Any]],
    task_label: str,
    doctor_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    filtered_points = [point for point in doctor_points if str(point.get("task_label") or "") == task_label]
    if not filtered_points:
        raise ValueError(f"医生性能 JSON 中未找到任务 {task_label} 的医生点。")

    if doctor_labels is None:
        return filtered_points

    requested_labels = [label for label in doctor_labels if label]
    requested_label_set = set(requested_labels)
    filtered_points = [point for point in filtered_points if str(point.get("doctor_label") or "") in requested_label_set]
    found_labels = {str(point.get("doctor_label") or "") for point in filtered_points}
    missing_labels = [label for label in requested_labels if label not in found_labels]
    if missing_labels:
        raise ValueError(f"任务 {task_label} 缺少指定医生点: {', '.join(missing_labels)}")
    return filtered_points


def _build_doctor_display_label(point: dict[str, Any]) -> str:
    return str(point.get("short_label") or point.get("doctor_label") or point.get("label") or "")


def _build_doctor_summary_lines(doctor_points: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for point in doctor_points:
        metrics = point.get("metrics") or {}
        accuracy = metrics.get("accuracy")
        short_label = str(point.get("short_label") or point.get("doctor_label") or point.get("label") or "")
        full_label = str(point.get("label") or "")
        task_label = str(point.get("task_label") or "")
        if accuracy is None:
            lines.append(f"{short_label}: {full_label} [{task_label}]")
        else:
            lines.append(f"{short_label}: {full_label} [{task_label}, Acc={float(accuracy):.4f}]")
    return lines


def _bbox_overlaps(box_a: Any, box_b: Any, padding: float = 4.0) -> bool:
    return not (
        box_a.x1 + padding < box_b.x0
        or box_a.x0 - padding > box_b.x1
        or box_a.y1 + padding < box_b.y0
        or box_a.y0 - padding > box_b.y1
    )


def _choose_annotation_layout(
    fig: Any,
    ax: Any,
    point: dict[str, Any],
    point_color: Any,
    label_text: str,
    all_points: list[dict[str, Any]],
    placed_bboxes: list[Any],
) -> tuple[tuple[int, int], str, str]:
    candidate_offsets: list[tuple[int, int]] = [
        (10, 10),
        (10, 26),
        (10, -12),
        (10, -28),
        (-10, 10),
        (-10, 26),
        (-10, -12),
        (-10, -28),
        (28, 0),
        (-28, 0),
        (42, 16),
        (-42, 16),
        (42, -16),
        (-42, -16),
    ]
    renderer = fig.canvas.get_renderer()
    other_point_pixels = [
        ax.transData.transform((float(other["fpr"]), float(other["tpr"])))
        for other in all_points
        if other is not point
    ]

    best_layout: tuple[tuple[int, int], str, str] | None = None
    best_score: float | None = None

    for xytext in candidate_offsets:
        ha = "left" if xytext[0] >= 0 else "right"
        va = "bottom" if xytext[1] >= 0 else "top"
        annotation = ax.annotate(
            label_text,
            (point["fpr"], point["tpr"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=10,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": point_color, "lw": 0.8, "alpha": 0.7},
            annotation_clip=False,
        )
        fig.canvas.draw()
        bbox = annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
        annotation.remove()

        overlap_penalty = sum(1 for existing_bbox in placed_bboxes if _bbox_overlaps(bbox, existing_bbox))
        point_cover_penalty = sum(1 for px, py in other_point_pixels if bbox.contains(px, py))

        axes_bbox = ax.get_window_extent(renderer=renderer)
        outside_penalty = 0
        if bbox.x0 < axes_bbox.x0:
            outside_penalty += axes_bbox.x0 - bbox.x0
        if bbox.x1 > axes_bbox.x1:
            outside_penalty += bbox.x1 - axes_bbox.x1
        if bbox.y0 < axes_bbox.y0:
            outside_penalty += axes_bbox.y0 - bbox.y0
        if bbox.y1 > axes_bbox.y1:
            outside_penalty += bbox.y1 - axes_bbox.y1

        distance_penalty = abs(xytext[0]) + abs(xytext[1]) * 0.8
        score = overlap_penalty * 10000 + point_cover_penalty * 1000 + outside_penalty * 10 + distance_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_layout = (xytext, ha, va)

    if best_layout is None:
        return (10, 10), "left", "bottom"
    return best_layout


def _plot_single_task_roc(
    curves: list[dict[str, Any]],
    doctor_points: list[dict[str, Any]],
    output_path: Path,
    title: str,
    doctor_point_color: Any = "black",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for curve in curves:
        fpr = np.asarray(curve["roc_curve_fpr"], dtype=np.float64)
        tpr = np.asarray(curve["roc_curve_tpr"], dtype=np.float64)
        auc_value = float(curve["roc_auc"])
        label = f"{curve['label']} (AUC={auc_value:.4f})"
        ax.plot(fpr, tpr, linewidth=2, label=label)

    fig.canvas.draw()
    placed_label_bboxes: list[Any] = []
    sorted_points = sorted(doctor_points, key=lambda item: (float(item["fpr"]), float(item["tpr"])))
    for point in sorted_points:
        ax.scatter(point["fpr"], point["tpr"], s=70, marker="o", color=doctor_point_color, zorder=5)
        label_text = _build_doctor_display_label(point)
        xytext, horizontal_alignment, vertical_alignment = _choose_annotation_layout(
            fig,
            ax,
            point,
            doctor_point_color,
            label_text,
            sorted_points,
            placed_label_bboxes,
        )
        annotation = ax.annotate(
            label_text,
            (point["fpr"], point["tpr"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=10,
            fontweight="bold",
            ha=horizontal_alignment,
            va=vertical_alignment,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": doctor_point_color, "linewidth": 0.8, "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": doctor_point_color, "lw": 0.8, "alpha": 0.7},
            annotation_clip=False,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_label_bboxes.append(annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.12))

    summary_lines = _build_doctor_summary_lines(sorted_points)
    if summary_lines:
        ax.text(
            0.015,
            0.985,
            "Doctor points\n" + "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "lightgray", "alpha": 0.92},
            zorder=6,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"单任务 AUROC 图已保存到: {output_path}")
    plt.close(fig)


def plot_single_task_auroc(
    input_paths: list[Path],
    labels: list[str],
    doctor_json_path: Path,
    task_label: str,
    output_path: Path,
    title: str,
    doctor_labels: list[str] | None = None,
    doctor_point_color: str = "black",
) -> None:
    if len(input_paths) != 2:
        raise ValueError("该脚本要求恰好传入两个 results JSON。")
    if len(labels) != 2:
        raise ValueError("该脚本要求恰好传入两个方法名称。")

    curves: list[dict[str, Any]] = []
    for path, label in zip(input_paths, labels):
        records = _load_results(path)
        payload = _resolve_roc_payload(records, path)
        payload["label"] = label
        payload["source"] = str(path)
        payload["task_label"] = task_label
        curves.append(payload)
        print(f"已加载曲线: {label} | schema={payload['schema']} | AUC={float(payload['roc_auc']):.4f}")

    loaded_points = _load_doctor_points(doctor_json_path)
    doctor_points = _filter_doctor_points_for_task(loaded_points, task_label, doctor_labels)
    for index, point in enumerate(
        sorted(doctor_points, key=lambda item: (str(item.get("doctor_label") or ""), str(item.get("label") or ""))),
        start=1,
    ):
        point["short_label"] = f"D{index}"
    print(f"已加载医生点: {len(doctor_points)} / {len(loaded_points)} | task={task_label} | source={doctor_json_path}")

    _plot_single_task_roc(curves, doctor_points, output_path, title, doctor_point_color)


def _default_output_path() -> Path:
    return Path.cwd() / "single_task_auroc.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制单个任务下两个方法的 AUROC 曲线，并叠加该任务的医生读片点")
    parser.add_argument("--inputs", nargs=2, required=True, help="两个 results_*.json 文件路径")
    parser.add_argument("--labels", nargs=2, required=True, help="两个方法名称，例如 ThyroidAgent BestSingleModel")
    parser.add_argument("--doctor-json", required=True, help="医生读片性能 JSON 路径，例如 doctor_multi_task_metrics.json")
    parser.add_argument("--task-label", required=True, help="医生性能 JSON 中对应的任务名，例如 BM")
    parser.add_argument("--doctor-labels", nargs="*", default=None, help="可选，指定要展示的医生标签列表")
    parser.add_argument("--doctor-color", default="black", help="医生点颜色，默认 black")
    parser.add_argument("--output", default=None, help="输出图片路径，可选；默认保存为当前运行目录下的 single_task_auroc.png")
    parser.add_argument("--title", default="Single-Task AUROC Comparison", help="图标题")
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"未找到 results JSON: {path}")

    doctor_json_path = Path(args.doctor_json).resolve()
    if not doctor_json_path.is_file():
        raise FileNotFoundError(f"未找到医生性能 JSON: {doctor_json_path}")

    output_path = Path(args.output).resolve() if args.output else _default_output_path()
    plot_single_task_auroc(
        input_paths=input_paths,
        labels=args.labels,
        doctor_json_path=doctor_json_path,
        task_label=args.task_label,
        output_path=output_path,
        title=args.title,
        doctor_labels=args.doctor_labels,
        doctor_point_color=args.doctor_color,
    )


if __name__ == "__main__":
    main()
