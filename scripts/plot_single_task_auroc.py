from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


NATURE_CURVE_COLORS = ("#0072B2", "#D55E00")
NATURE_RANDOM_COLOR = "#BFBFBF"
NATURE_DOCTOR_COLOR = "#222222"


def _apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


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


def _bbox_overlaps(box_a: Any, box_b: Any, padding: float = 2.0) -> bool:
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
        (5, 5),
        (5, 11),
        (5, -7),
        (5, -13),
        (-5, 5),
        (-5, 11),
        (-5, -7),
        (-5, -13),
        (12, 0),
        (-12, 0),
        (18, 8),
        (-18, 8),
        (18, -8),
        (-18, -8),
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
            fontsize=7,
            ha=ha,
            va=va,
            color=point_color,
            bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": point_color, "lw": 0.5, "alpha": 0.7},
            annotation_clip=False,
        )
        fig.canvas.draw()
        bbox = annotation.get_window_extent(renderer=renderer).expanded(1.03, 1.08)
        annotation.remove()

        overlap_penalty = sum(1 for existing_bbox in placed_bboxes if _bbox_overlaps(bbox, existing_bbox))
        point_cover_penalty = sum(1 for px, py in other_point_pixels if bbox.contains(px, py))

        axes_bbox = ax.get_window_extent(renderer=renderer)
        outside_penalty = 0.0
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
        return (5, 5), "left", "bottom"
    return best_layout


def _resolve_curve_color(index: int) -> str:
    if index < len(NATURE_CURVE_COLORS):
        return NATURE_CURVE_COLORS[index]
    return NATURE_CURVE_COLORS[index % len(NATURE_CURVE_COLORS)]


def _plot_single_task_roc(
    curves: list[dict[str, Any]],
    doctor_points: list[dict[str, Any]],
    output_path: Path,
    title: str,
    doctor_point_color: Any = NATURE_DOCTOR_COLOR,
) -> None:
    _apply_nature_style()

    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    ax.set_box_aspect(1)

    for index, curve in enumerate(curves):
        fpr = np.asarray(curve["roc_curve_fpr"], dtype=np.float64)
        tpr = np.asarray(curve["roc_curve_tpr"], dtype=np.float64)
        auc_value = float(curve["roc_auc"])
        curve_color = _resolve_curve_color(index)
        label = f"{curve['label']} (AUC {auc_value:.3f})"
        ax.plot(fpr, tpr, color=curve_color, linewidth=1.8, label=label, solid_capstyle="round")

    fig.canvas.draw()
    placed_label_bboxes: list[Any] = []
    sorted_points = sorted(doctor_points, key=lambda item: (float(item["fpr"]), float(item["tpr"])))
    for point in sorted_points:
        ax.scatter(
            point["fpr"],
            point["tpr"],
            s=18,
            marker="o",
            facecolor=doctor_point_color,
            edgecolor=doctor_point_color,
            linewidth=0.5,
            zorder=5,
        )
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
            fontsize=7,
            ha=horizontal_alignment,
            va=vertical_alignment,
            color=doctor_point_color,
            bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": doctor_point_color, "lw": 0.5, "alpha": 0.7},
            annotation_clip=False,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_label_bboxes.append(annotation.get_window_extent(renderer=renderer).expanded(1.03, 1.08))

    ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), linewidth=0.8, color=NATURE_RANDOM_COLOR, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title, pad=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", length=3, width=0.8)
    ax.legend(loc="lower right", frameon=False, handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["dpi"] = 600
    fig.savefig(output_path, **save_kwargs)
    print(f"单任务 AUROC 图已保存到: {output_path}")
    plt.close(fig)


def plot_single_task_auroc(
    input_paths: list[Path],
    labels: list[str],
    output_path: Path,
    title: str,
    doctor_json_path: Path | None = None,
    task_label: str | None = None,
    doctor_labels: list[str] | None = None,
    doctor_point_color: str = NATURE_DOCTOR_COLOR,
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

    doctor_points: list[dict[str, Any]] = []
    if doctor_json_path is not None:
        if not task_label:
            raise ValueError("传入 --doctor-json 时必须同时传入 --task-label。")
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
    return Path.cwd() / "single_task_auroc.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制单个任务下两个方法的 AUROC 曲线，并可选叠加该任务的医生读片点")
    parser.add_argument("--inputs", nargs=2, required=True, help="两个 results_*.json 文件路径")
    parser.add_argument("--labels", nargs=2, required=True, help="两个方法名称，例如 ThyroidAgent BestSingleModel")
    parser.add_argument("--doctor-json", default=None, help="医生读片性能 JSON 路径，可选，例如 doctor_multi_task_metrics.json")
    parser.add_argument("--task-label", default=None, help="医生性能 JSON 中对应的任务名；传入 --doctor-json 时必填，例如 BM")
    parser.add_argument("--doctor-labels", nargs="*", default=None, help="可选，指定要展示的医生标签列表")
    parser.add_argument("--doctor-color", default=NATURE_DOCTOR_COLOR, help="医生点颜色，默认使用接近 Nature 风格的深灰色")
    parser.add_argument("--output", default=None, help="输出图片路径，可选；默认保存为当前运行目录下的 single_task_auroc.pdf")
    parser.add_argument("--title", default="", help="图标题；默认不显示标题以贴近期刊主图风格")
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"未找到 results JSON: {path}")

    doctor_json_path = Path(args.doctor_json).resolve() if args.doctor_json else None
    if doctor_json_path is not None and not doctor_json_path.is_file():
        raise FileNotFoundError(f"未找到医生性能 JSON: {doctor_json_path}")

    output_path = Path(args.output).resolve() if args.output else _default_output_path()
    plot_single_task_auroc(
        input_paths=input_paths,
        labels=args.labels,
        output_path=output_path,
        title=args.title,
        doctor_json_path=doctor_json_path,
        task_label=args.task_label,
        doctor_labels=args.doctor_labels,
        doctor_point_color=args.doctor_color,
    )


if __name__ == "__main__":
    main()
