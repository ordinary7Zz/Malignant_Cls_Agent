from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve


NATURE_CURVE_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#9C755F",
)
NATURE_RANDOM_COLOR = "#BFBFBF"
NATURE_DOCTOR_COLOR = "#222222"
NATURE_DOCTOR_MARKER_FACE = "#F5B21A"
NATURE_DOCTOR_MARKERS = ("^", "v")


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


def _compute_prc_from_scores(y_true_arr: np.ndarray, y_prob_arr: np.ndarray) -> tuple[list[float], list[float], list[float], float]:
    pos_total = int(np.sum(y_true_arr == 1))
    neg_total = int(np.sum(y_true_arr == 0))
    if pos_total == 0 or neg_total == 0:
        raise ValueError("真实标签只有一个类别，无法绘制 AUPRC 曲线。")

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    auprc_value = float(average_precision_score(y_true_arr, y_prob_arr))
    return precision.tolist(), recall.tolist(), thresholds.tolist(), auprc_value


def _coerce_binary_arrays(y_true: list[int], y_prob: list[float], source: Path) -> tuple[np.ndarray, np.ndarray]:
    if not y_true:
        raise ValueError(f"未在 results JSON 中找到可用于绘制 AUPRC 的样本字段: {source}")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError(f"真实标签只有一个类别，无法绘制 AUPRC 曲线: {source}")
    return y_true_arr, y_prob_arr


def _build_prc_from_current_schema(records: list[dict[str, Any]], source: Path) -> dict[str, Any] | None:
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
    precision, recall, thresholds, auprc_value = _compute_prc_from_scores(y_true_arr, y_prob_arr)
    positive_rate = float(np.mean(y_true_arr == 1))
    return {
        "schema": "current",
        "pr_curve_precision": precision,
        "pr_curve_recall": recall,
        "pr_curve_thresholds": thresholds,
        "auprc": auprc_value,
        "positive_rate": positive_rate,
        "n_samples": int(y_true_arr.shape[0]),
    }


def _build_prc_from_classification_agent_schema(records: list[dict[str, Any]], source: Path) -> dict[str, Any] | None:
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
    precision, recall, thresholds, auprc_value = _compute_prc_from_scores(y_true_arr, y_prob_arr)
    positive_rate = float(np.mean(y_true_arr == 1))
    return {
        "schema": "classification_agent",
        "pr_curve_precision": precision,
        "pr_curve_recall": recall,
        "pr_curve_thresholds": thresholds,
        "auprc": auprc_value,
        "positive_rate": positive_rate,
        "n_samples": int(y_true_arr.shape[0]),
    }


def _resolve_prc_payload(records: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    payload = _build_prc_from_current_schema(records, source)
    if payload is not None:
        return payload

    payload = _build_prc_from_classification_agent_schema(records, source)
    if payload is not None:
        return payload

    raise ValueError(f"无法识别或提取可用于 AUPRC 的字段: {source}")


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
        metrics = item.get("metrics") or {}
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        if precision is None or recall is None:
            confusion_matrix = item.get("confusion_matrix") or {}
            tp = confusion_matrix.get("tp")
            fp = confusion_matrix.get("fp")
            fn = confusion_matrix.get("fn")
            if precision is None and tp is not None and fp is not None and (tp + fp) > 0:
                precision = float(tp) / float(tp + fp)
            if recall is None and tp is not None and fn is not None and (tp + fn) > 0:
                recall = float(tp) / float(tp + fn)
        if precision is None or recall is None:
            continue

        task_label = str(item.get("task_label") or "")
        doctor_label = str(item.get("doctor_label") or "")
        points.append(
            {
                "task_label": task_label,
                "doctor_label": doctor_label,
                "label": str(item.get("label") or f"{task_label} - {doctor_label}"),
                "precision": float(precision),
                "recall": float(recall),
                "n_samples": item.get("n_samples"),
                "metrics": metrics,
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


def _resolve_curve_color(index: int) -> str:
    if index < len(NATURE_CURVE_COLORS):
        return NATURE_CURVE_COLORS[index]
    return NATURE_CURVE_COLORS[index % len(NATURE_CURVE_COLORS)]


def _resolve_curve_alpha(auprc_value: float, min_auprc: float, max_auprc: float) -> float:
    if max_auprc <= min_auprc:
        return 0.9
    normalized = (float(auprc_value) - min_auprc) / (max_auprc - min_auprc)
    return float(np.clip(0.22 + (normalized**1.4) * 0.78, 0.22, 1.0))


def _resolve_curve_linewidth(auprc_value: float, min_auprc: float, max_auprc: float) -> float:
    if max_auprc <= min_auprc:
        return 1.8
    normalized = (float(auprc_value) - min_auprc) / (max_auprc - min_auprc)
    return float(np.clip(1.15 + normalized * 0.95, 1.15, 2.1))


def _resolve_doctor_marker(index: int) -> str:
    if index < len(NATURE_DOCTOR_MARKERS):
        return NATURE_DOCTOR_MARKERS[index]
    return NATURE_DOCTOR_MARKERS[index % len(NATURE_DOCTOR_MARKERS)]


def _build_doctor_legend_label(point: dict[str, Any], index: int) -> str:
    doctor_label = str(point.get("doctor_label") or "").strip()
    if doctor_label:
        return doctor_label.replace("_", " ").title()
    return f"Doctor {index + 1}"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("--curve-alpha 只能传 true 或 false")


def _plot_single_task_prc(
    curves: list[dict[str, Any]],
    doctor_points: list[dict[str, Any]],
    output_path: Path,
    title: str,
    doctor_point_color: Any = NATURE_DOCTOR_COLOR,
    use_curve_alpha: bool = True,
) -> None:
    _apply_nature_style()

    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    ax.set_box_aspect(1)

    auprc_values = [float(curve["auprc"]) for curve in curves]
    min_auprc = min(auprc_values)
    max_auprc = max(auprc_values)

    curves_sorted = sorted(curves, key=lambda item: float(item["auprc"]), reverse=True)

    for index, curve in enumerate(curves_sorted):
        precision = np.asarray(curve["pr_curve_precision"], dtype=np.float64)
        recall = np.asarray(curve["pr_curve_recall"], dtype=np.float64)
        auprc_value = float(curve["auprc"])
        curve_color = _resolve_curve_color(index)
        curve_alpha = _resolve_curve_alpha(auprc_value, min_auprc, max_auprc)
        curve_linewidth = _resolve_curve_linewidth(auprc_value, min_auprc, max_auprc)
        label = f"{curve['label']} ({auprc_value:.4f})"
        plot_kwargs: dict[str, Any] = {
            "color": curve_color,
            "linewidth": curve_linewidth,
            "label": label,
            "solid_capstyle": "round",
        }
        if use_curve_alpha:
            plot_kwargs["alpha"] = curve_alpha
        ax.plot(recall[::-1], precision[::-1], **plot_kwargs)

    fig.canvas.draw()
    sorted_points = sorted(doctor_points, key=lambda item: (float(item["recall"]), float(item["precision"])))
    doctor_handles: list[Line2D] = []
    for index, point in enumerate(sorted_points):
        marker = _resolve_doctor_marker(index)
        legend_label = _build_doctor_legend_label(point, index)
        ax.scatter(
            point["recall"],
            point["precision"],
            s=60,
            marker=marker,
            facecolor=NATURE_DOCTOR_MARKER_FACE,
            edgecolor=doctor_point_color,
            linewidth=0.7,
            zorder=5,
        )
        doctor_handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker=marker,
                markersize=7.5,
                markerfacecolor=NATURE_DOCTOR_MARKER_FACE,
                markeredgecolor=doctor_point_color,
                markeredgewidth=0.7,
                label=legend_label,
            )
        )

    handles, labels = ax.get_legend_handles_labels()
    plot_handles: list[Any] = []
    plot_labels: list[str] = []
    random_handle: Any | None = None
    for handle, label in zip(handles, labels):
        if label == "Random":
            random_handle = handle
        else:
            plot_handles.append(handle)
            plot_labels.append(label)

    legend_handles = plot_handles + doctor_handles
    legend_labels = plot_labels + [handle.get_label() for handle in doctor_handles]
    if random_handle is not None:
        legend_handles.append(random_handle)
        legend_labels.append("Random")

    ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), linewidth=0.8, color=NATURE_RANDOM_COLOR, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if title:
        ax.set_title(title, pad=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", length=3, width=0.8)
    ax.legend(legend_handles, legend_labels, loc="lower right", frameon=False, handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["dpi"] = 600
    fig.savefig(output_path, **save_kwargs)
    print(f"单任务 AUPRC 图已保存到: {output_path}")
    plt.close(fig)


def plot_single_task_auprc(
    input_paths: list[Path],
    labels: list[str],
    output_path: Path,
    title: str,
    doctor_json_path: Path | None = None,
    task_label: str | None = None,
    doctor_labels: list[str] | None = None,
    doctor_point_color: str = NATURE_DOCTOR_COLOR,
    use_curve_alpha: bool = True,
) -> None:
    if not input_paths:
        raise ValueError("至少需要传入一个 results JSON。")
    if len(input_paths) != len(labels):
        raise ValueError("results JSON 的数量必须与方法名称数量一致。")

    curves: list[dict[str, Any]] = []
    for path, label in zip(input_paths, labels):
        records = _load_results(path)
        payload = _resolve_prc_payload(records, path)
        payload["label"] = label
        payload["source"] = str(path)
        payload["task_label"] = task_label
        curves.append(payload)
        print(f"已加载曲线: {label} | schema={payload['schema']} | AUPRC={float(payload['auprc']):.4f}")

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

    _plot_single_task_prc(curves, doctor_points, output_path, title, doctor_point_color, use_curve_alpha)


def _default_output_path() -> Path:
    return Path.cwd() / "single_task_auprc.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制单个任务下一个或多个方法的 AUPRC 曲线，并可选叠加该任务的医生读片点")
    parser.add_argument("--inputs", nargs="*", required=True, action="extend", help="一个或多个 results_*.json 文件路径")
    parser.add_argument("--labels", nargs="*", required=True, action="extend", help="与输入文件一一对应的方法名称列表")
    parser.add_argument("--doctor-json", default=None, help="医生读片性能 JSON 路径，可选，例如 doctor_multi_task_metrics.json")
    parser.add_argument("--task-label", default=None, help="医生性能 JSON 中对应的任务名；传入 --doctor-json 时必填，例如 BM")
    parser.add_argument("--doctor-labels", nargs="*", default=None, help="可选，指定要展示的医生标签列表")
    parser.add_argument("--doctor-color", default=NATURE_DOCTOR_COLOR, help="医生点颜色，默认使用接近 Nature 风格的深灰色")
    parser.add_argument(
        "--curve-alpha",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="是否按 AUPRC 动态设置曲线透明度，传 true 或 false；默认 true",
    )
    parser.add_argument("--output", default=None, help="输出图片路径，可选；默认保存为当前运行目录下的 single_task_auprc.pdf")
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
    plot_single_task_auprc(
        input_paths=input_paths,
        labels=args.labels,
        output_path=output_path,
        title=args.title,
        doctor_json_path=doctor_json_path,
        task_label=args.task_label,
        doctor_labels=args.doctor_labels,
        doctor_point_color=args.doctor_color,
        use_curve_alpha=args.curve_alpha,
    )


if __name__ == "__main__":
    main()
