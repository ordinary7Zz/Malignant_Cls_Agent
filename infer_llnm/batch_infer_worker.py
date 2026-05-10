"""
在独立子进程中批量运行 LLNM 推理，避免与项目根目录 ``models`` 包命名冲突。

由 ``auxiliary_binary_inference`` 以 ``python -m infer_llnm.batch_infer_worker``
或 ``python infer_llnm/batch_infer_worker.py`` 调用；从 stdin 读 JSON，向 stdout 写 JSON。

stdin JSON 字段:
  - model_path, batch_size, num_classes, device, threshold
  - norm_params_file (可选)
  - default_report, default_age, default_sex, default_shape_echo [a,b]
  - records: [ {"path": "绝对路径", "out_key": "输出键"}, ... ]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    # 兼容两种启动方式：
    # 1) 在 Malignant_Cls_Agent 目录下直接执行脚本
    # 2) 作为包模块执行（python -m infer_llnm.batch_infer_worker）
    # 为绝对导入 `Malignant_Cls_Agent.*` 补充包上级目录到 sys.path。
    here_path = Path(__file__).resolve()
    infer_llnm_dir = str(here_path.parent)
    package_dir = here_path.parents[1]  # .../Malignant_Cls_Agent
    package_parent_dir = str(package_dir.parent)  # .../Classification_Agent

    if infer_llnm_dir not in sys.path:
        sys.path.insert(0, infer_llnm_dir)
    if package_parent_dir not in sys.path:
        sys.path.insert(0, package_parent_dir)

    payload = json.load(sys.stdin)
    records = payload["records"]
    for r in records:
        r["path"] = Path(r["path"])

    from torchvision import transforms

    import Malignant_Cls_Agent.infer_llnm.infer_images as llnm_infer

    ImageRecordsInferenceDataset = llnm_infer.ImageRecordsInferenceDataset
    run_inference = llnm_infer.run_inference

    shape_echo = payload.get("default_shape_echo", [0.0, 0.0])
    shape_tuple = (float(shape_echo[0]), float(shape_echo[1]))

    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageRecordsInferenceDataset(
        records,
        data_transforms,
        default_report=str(payload.get("default_report", "") or ""),
        default_age=float(payload.get("default_age", 50.0)),
        default_sex=float(payload.get("default_sex", 1.0)),
        default_shape_echo=shape_tuple,
        norm_params_file=payload.get("norm_params_file"),
    )

    rels, probs = run_inference(
        model_path=payload["model_path"],
        dataset=dataset,
        batch_size=int(payload.get("batch_size", 4)),
        num_classes=payload.get("num_classes"),
        device=str(payload.get("device", "cuda")),
    )

    threshold = float(payload.get("threshold", 0.5))
    out_rows = []
    n = min(len(rels), len(probs))
    for i in range(n):
        row = probs[i]
        if row.shape[0] == 1:
            prob_1 = float(row[0])
            prob_0 = 1.0 - prob_1
            pred = int(prob_1 >= threshold)
        else:
            pred = int(np.argmax(row))
            if row.shape[0] == 2:
                pred = int(row[1] >= threshold)
            prob_0 = float(row[0]) if row.shape[0] > 0 else 0.0
            prob_1 = float(row[1]) if row.shape[0] > 1 else float(row[0])
        out_rows.append(
            {
                "path": str(records[i]["path"].resolve()),
                "relative_path": rels[i],
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
                "pred_class": pred,
            }
        )

    json.dump({"ok": True, "results": out_rows}, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"ok": False, "error": str(e)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.exit(1)
