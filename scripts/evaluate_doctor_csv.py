from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if any(cell.strip() for cell in row)]
            if not rows:
                raise ValueError(f"CSV 文件为空: {path}")
            header = [cell.strip() for cell in rows[0]]
            data_rows = rows[1:]
            return header, data_rows
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"无法读取 CSV 文件: {path}")


def _parse_binary_label(value: str, row_index: int, column_name: str) -> int:
    text = value.strip()
    if text not in {"0", "1"}:
        raise ValueError(f"第 {row_index} 行的 {column_name} 不是 0/1: {value!r}")
    return int(text)


def _make_unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for index, label in enumerate(labels, start=1):
        base = label.strip() or f"doctor_{index}"
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        unique_labels.append(base if seen == 0 else f"{base}_{seen + 1}")
    return unique_labels


def _load_task_annotations(
    path: Path,
    task_label: str,
    filename_col: int,
    label_col: int,
    doctor_start_col: int,
) -> dict[str, Any]:
    header, data_rows = _read_csv_rows(path)
    min_required_cols = max(filename_col, label_col, doctor_start_col)
    if len(header) <= min_required_cols:
        raise ValueError(
            f"CSV 列数不足，至少需要 {min_required_cols + 1} 列，当前表头为: {header}"
        )

    doctor_column_indices = list(range(doctor_start_col, len(header)))
    if not doctor_column_indices:
        raise ValueError(f"未找到医生分类列: {path}")

    doctor_header_labels = [header[index] for index in doctor_column_indices]
    doctor_labels = _make_unique_labels(doctor_header_labels)

    records: list[dict[str, Any]] = []
    for data_offset, row in enumerate(data_rows, start=2):
        if len(row) <= doctor_column_indices[-1]:
            raise ValueError(f"第 {data_offset} 行列数不足，至少需要 {doctor_column_indices[-1] + 1} 列: {row}")

        filename = row[filename_col].strip()
        if not filename:
            raise ValueError(f"第 {data_offset} 行 filename 为空。")

        true_label = _parse_binary_label(row[label_col], data_offset, "实际 label")
        doctor_predictions: dict[str, int] = {}
        for doctor_label, column_index in zip(doctor_labels, doctor_column_indices):
            doctor_predictions[doctor_label] = _parse_binary_label(row[column_index], data_offset, doctor_label)

        records.append(
            {
                "filename": filename,
                "true_label": true_label,
                "doctor_predictions": doctor_predictions,
            }
        )

    if not records:
        raise ValueError(f"CSV 中没有有效数据行: {path}")

    return {
        "task_label": task_label,
        "input_csv": str(path),
        "header": header,
        "doctor_columns": [
            {
                "doctor_label": doctor_label,
                "column_name": header[column_index],
                "column_index": column_index,
            }
            for doctor_label, column_index in zip(doctor_labels, doctor_column_indices)
        ],
        "records": records,
    }


def _compute_confusion_matrix(records: list[dict[str, Any]], doctor_label: str) -> dict[str, int]:
    tp = sum(
        1
        for item in records
        if item["doctor_predictions"][doctor_label] == 1 and item["true_label"] == 1
    )
    tn = sum(
        1
        for item in records
        if item["doctor_predictions"][doctor_label] == 0 and item["true_label"] == 0
    )
    fp = sum(
        1
        for item in records
        if item["doctor_predictions"][doctor_label] == 1 and item["true_label"] == 0
    )
    fn = sum(
        1
        for item in records
        if item["doctor_predictions"][doctor_label] == 0 and item["true_label"] == 1
    )
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _compute_metrics(confusion: dict[str, int]) -> dict[str, float | int | None]:
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    total = tp + tn + fp + fn

    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)

    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "n_samples": total,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": _safe_div(fp, fp + tn),
        "f1": f1,
    }


def _build_doctor_report(task_payload: dict[str, Any], doctor_info: dict[str, Any]) -> dict[str, Any]:
    records = task_payload["records"]
    confusion = _compute_confusion_matrix(records, doctor_info["doctor_label"])
    metrics = _compute_metrics(confusion)
    return {
        "doctor_label": doctor_info["doctor_label"],
        "column_name": doctor_info["column_name"],
        "column_index": doctor_info["column_index"],
        "n_samples": metrics["n_samples"],
        "n_positive": metrics["n_positive"],
        "n_negative": metrics["n_negative"],
        "confusion_matrix": confusion,
        "metrics": metrics,
        "roc_point": {
            "fpr": metrics["fpr"],
            "tpr": metrics["recall"],
        },
    }


def _build_task_report(task_payload: dict[str, Any]) -> dict[str, Any]:
    doctors = [_build_doctor_report(task_payload, doctor_info) for doctor_info in task_payload["doctor_columns"]]
    if not doctors:
        raise ValueError(f"任务 {task_payload['task_label']} 没有医生结果可汇总。")

    return {
        "task_label": task_payload["task_label"],
        "input_csv": task_payload["input_csv"],
        "n_doctors": len(doctors),
        "n_samples": doctors[0]["n_samples"],
        "n_positive": doctors[0]["n_positive"],
        "n_negative": doctors[0]["n_negative"],
        "doctors": doctors,
    }


def _build_report(task_reports: list[dict[str, Any]]) -> dict[str, Any]:
    plot_points: list[dict[str, Any]] = []
    for task in task_reports:
        for doctor in task["doctors"]:
            plot_points.append(
                {
                    "task_label": task["task_label"],
                    "doctor_label": doctor["doctor_label"],
                    "label": f"{task['task_label']} - {doctor['doctor_label']}",
                    "input_csv": task["input_csv"],
                    "n_samples": doctor["n_samples"],
                    "confusion_matrix": doctor["confusion_matrix"],
                    "metrics": doctor["metrics"],
                    "roc_point": doctor["roc_point"],
                }
            )

    return {
        "record_type": "doctor_multi_task_roc_point_summary",
        "schema": "doctor_csv_multi_task_multi_doctor",
        "n_tasks": len(task_reports),
        "tasks": task_reports,
        "plot_points": plot_points,
    }


def _default_output_path() -> Path:
    return Path.cwd() / "doctor_multi_task_metrics.json"


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _print_report(report: dict[str, Any]) -> None:
    print(f"任务数: {report['n_tasks']}")
    for task in report["tasks"]:
        print(f"任务: {task['task_label']} | 样本数={task['n_samples']} | 医生数={task['n_doctors']}")
        for doctor in task["doctors"]:
            confusion = doctor["confusion_matrix"]
            metrics = doctor["metrics"]
            roc_point = doctor["roc_point"]
            print(f"  医生: {doctor['doctor_label']}")
            print(f"    混淆矩阵: TP={confusion['tp']} FP={confusion['fp']} FN={confusion['fn']} TN={confusion['tn']}")
            print(
                "    指标: "
                f"acc={_format_metric(metrics['accuracy'])} "
                f"precision={_format_metric(metrics['precision'])} "
                f"recall={_format_metric(metrics['recall'])} "
                f"specificity={_format_metric(metrics['specificity'])} "
                f"fpr={_format_metric(metrics['fpr'])} "
                f"f1={_format_metric(metrics['f1'])}"
            )
            print(
                "    ROC 点: "
                f"(FPR={_format_metric(roc_point['fpr'])}, TPR={_format_metric(roc_point['tpr'])})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="根据多个任务的医生读片 CSV 计算多医生性能指标和混淆矩阵")
    parser.add_argument("--inputs", nargs="+", required=True, help="一个或多个任务 CSV 路径")
    parser.add_argument("--task-labels", nargs="*", default=None, help="每个 CSV 对应的任务名，如 BM FTCPTC LNMCN01")
    parser.add_argument("--output", default=None, help="输出 JSON 路径，可选；默认保存为当前运行目录下的 doctor_multi_task_metrics.json")
    parser.add_argument("--filename-col", type=int, default=0, help="filename 所在列索引，默认 0")
    parser.add_argument("--label-col", type=int, default=1, help="实际 label 所在列索引，默认 1")
    parser.add_argument("--doctor-start-col", type=int, default=2, help="医生分类起始列索引，默认 2")
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"未找到 CSV 文件: {path}")

    if args.task_labels is not None and len(args.task_labels) > 0 and len(args.task_labels) != len(input_paths):
        raise ValueError("--task-labels 的数量必须与 --inputs 一致。")

    task_labels = args.task_labels if args.task_labels else [path.stem for path in input_paths]
    output_path = Path(args.output).resolve() if args.output else _default_output_path()

    task_reports: list[dict[str, Any]] = []
    for input_path, task_label in zip(input_paths, task_labels):
        task_payload = _load_task_annotations(
            input_path,
            task_label=task_label,
            filename_col=args.filename_col,
            label_col=args.label_col,
            doctor_start_col=args.doctor_start_col,
        )
        task_reports.append(_build_task_report(task_payload))

    report = _build_report(task_reports)
    _print_report(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
