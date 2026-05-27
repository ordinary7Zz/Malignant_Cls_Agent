# 给大模型的任务提示

你现在的任务是：修改某个分类模型的推理/评估代码，使它最终导出的结果 JSON 符合本文档定义的格式要求。

目标是把不同模型的分类推理结果统一成可被 `scripts/plot_single_task_auroc.py` 直接读取的标准格式，方便后续对多个模型绘制 AUROC 对比曲线图。

请注意：

- 你需要优先修改“模型推理结果导出”相关代码，而不是绘图代码。
- 你的输出目标是统一分类结果格式，不是改变模型本身的训练逻辑。
- 如果原始代码已经能得到逐样本预测结果，请尽量整理并导出为本文档推荐的统一 JSON。
- 除非有充分理由，否则优先输出本文档中的 **格式 B（`true_label + prob_class_1`）**，因为它最通用、最适合多模型统一比较。
- 如果原始代码已经直接计算好了 ROC 曲线和 AUC，也可以输出为 **格式 A（`roc_summary`）**。
- 修改后应确保每个模型最终都能产出一个合法 JSON 文件，供 `scripts/plot_single_task_auroc.py --inputs ...` 直接使用。

下面是 `plot_single_task_auroc.py` 当前支持的 JSON 输入格式说明，请严格按照其中一种格式修改对应模型代码。

# `plot_single_task_auroc.py` 支持的 JSON 格式模板

对应脚本：`scripts/plot_single_task_auroc.py`

这个脚本的 `--inputs` 参数要求传入 **一个或多个 results JSON 文件**。每个文件都代表一个模型的结果，然后通过 `--labels` 给每个文件指定显示名称。

脚本当前支持 **3 种 results JSON 格式**，按识别优先级如下：

1. `roc_summary` 格式
2. `current schema`（样本级 `true_label + prob_class_1`）
3. `classification_agent schema`（样本级 `ground_truth_label + malignant_probability`）

如果一个文件里同时出现多种信息，脚本会优先使用最后一个 `record_type == "roc_summary"` 的记录。

---

## 最推荐：格式 A（`roc_summary`）

适合场景：
- 你已经提前算好了 ROC 曲线和 AUC
- 不想暴露逐样本预测
- 只想让绘图脚本直接画图

### 最小可用模板

```json
[
  {
    "record_type": "roc_summary",
    "roc_curve_fpr": [0.0, 0.05, 0.12, 0.30, 1.0],
    "roc_curve_tpr": [0.0, 0.62, 0.81, 0.93, 1.0],
    "roc_auc": 0.9123
  }
]
```

### 完整模板

```json
[
  {
    "record_type": "roc_summary",
    "roc_curve_fpr": [0.0, 0.05, 0.12, 0.30, 1.0],
    "roc_curve_tpr": [0.0, 0.62, 0.81, 0.93, 1.0],
    "roc_curve_thresholds": [1.1, 0.91, 0.73, 0.40, 0.0],
    "roc_auc": 0.9123,
    "n_aligned_samples": 500
  }
]
```

### 必填字段

- `record_type`: 必须为 `"roc_summary"`
- `roc_curve_fpr`: 假阳性率数组
- `roc_curve_tpr`: 真阳性率数组
- `roc_auc`: AUC 数值

### 选填字段

- `roc_curve_thresholds`: 阈值数组
- `n_aligned_samples`: 样本数

### 注意事项

- `roc_curve_fpr` 和 `roc_curve_tpr` 长度应一致。
- 建议 `roc_curve_fpr` 从 `0` 开始到 `1` 结束，`roc_curve_tpr` 也对应完整曲线。
- 如果你只是为了多个模型画对比图，这是最省事的统一格式。

---

## 推荐：格式 B（`current schema`，样本级）

适合场景：
- 你有每个样本的真实标签和模型输出概率
- 希望脚本自动重新计算 ROC 和 AUC
- 希望以后还能复用这些结果做别的分析

脚本会遍历数组中的每一条记录，读取：

- `true_label`
- `prob_class_1`

并要求两类标签都存在，否则无法计算 AUROC。

### 最小可用模板

```json
[
  {
    "record_type": "sample",
    "true_label": 1,
    "prob_class_1": 0.93
  },
  {
    "record_type": "sample",
    "true_label": 0,
    "prob_class_1": 0.21
  },
  {
    "record_type": "sample",
    "true_label": 1,
    "prob_class_1": 0.84
  },
  {
    "record_type": "sample",
    "true_label": 0,
    "prob_class_1": 0.12
  }
]
```

### 更接近项目现有输出的模板

```json
[
  {
    "record_type": "sample",
    "image_file": "D:/data/case_001.png",
    "image_name": "case_001.png",
    "selected_model": "model_a",
    "predicted_class": 1,
    "confidence": 0.93,
    "prob_class_0": 0.07,
    "prob_class_1": 0.93,
    "true_label": 1,
    "reasoning": "optional"
  },
  {
    "record_type": "sample",
    "image_file": "D:/data/case_002.png",
    "image_name": "case_002.png",
    "selected_model": "model_a",
    "predicted_class": 0,
    "confidence": 0.79,
    "prob_class_0": 0.79,
    "prob_class_1": 0.21,
    "true_label": 0,
    "reasoning": "optional"
  }
]
```

### 实际必填字段

每条样本记录至少需要：

- `true_label`: 真实标签，二分类，通常取 `0/1`
- `prob_class_1`: 样本属于正类（malignant）的概率，范围建议为 `[0, 1]`

### 可选字段

这些字段当前绘图脚本不会强依赖，但保留会更完整：

- `record_type`，建议写成 `"sample"`
- `image_file`
- `image_name`
- `selected_model`
- `predicted_class`
- `confidence`
- `prob_class_0`
- `reasoning`

### 注意事项

- JSON 顶层必须是 **列表**。
- 至少要有一个正样本和一个负样本，否则脚本会报错。
- 如果某条记录缺少 `true_label` 或 `prob_class_1`，该条会被跳过。
- 这是最适合多个模型统一导出的格式之一，因为每个模型都只需要输出“真实标签 + 正类概率”。

---

## 兼容：格式 C（`classification_agent schema`）

适合场景：
- 你已有旧版结果文件
- 字段名已经固定为 `ground_truth_label` 和 `malignant_probability`

### 最小可用模板

```json
[
  {
    "ground_truth_label": 1,
    "malignant_probability": 0.93
  },
  {
    "ground_truth_label": 0,
    "malignant_probability": 0.21
  },
  {
    "ground_truth_label": 1,
    "malignant_probability": 0.84
  },
  {
    "ground_truth_label": 0,
    "malignant_probability": 0.12
  }
]
```

### 实际必填字段

- `ground_truth_label`
- `malignant_probability`

### 注意事项

- 同样要求顶层是 **列表**。
- 同样要求正负样本都存在。
- 如果你现在要统一多个模型输出，除非历史包袱很重，否则更建议统一成 **格式 A** 或 **格式 B**。

---

## 多模型对比时建议统一成什么格式？

### 建议 1：如果你已经有 ROC 曲线数组，统一成格式 A

每个模型输出一个 JSON，例如：

- `model_resnet.json`
- `model_vit.json`
- `model_swin.json`

每个文件内容都写成：

```json
[
  {
    "record_type": "roc_summary",
    "roc_curve_fpr": [0.0, 0.05, 0.12, 0.30, 1.0],
    "roc_curve_tpr": [0.0, 0.62, 0.81, 0.93, 1.0],
    "roc_auc": 0.9123
  }
]
```

这是最直接的多模型对比方案。

### 建议 2：如果你有逐样本预测概率，统一成格式 B

每个模型输出同一批样本的：

- `true_label`
- `prob_class_1`

例如：

```json
[
  {"record_type": "sample", "sample_id": "case_001", "true_label": 1, "prob_class_1": 0.93},
  {"record_type": "sample", "sample_id": "case_002", "true_label": 0, "prob_class_1": 0.21},
  {"record_type": "sample", "sample_id": "case_003", "true_label": 1, "prob_class_1": 0.84},
  {"record_type": "sample", "sample_id": "case_004", "true_label": 0, "prob_class_1": 0.12}
]
```

这是最通用的统一方案。

---

## 命令行示例

假设你准备了 3 个模型文件：

- `model_resnet.json`
- `model_vit.json`
- `model_swin.json`

那么可以这样画图：

```bash
python scripts/plot_single_task_auroc.py \
  --inputs model_resnet.json model_vit.json model_swin.json \
  --labels ResNet ViT Swin \
  --output single_task_auroc.pdf
```

Windows PowerShell 里可以写成单行：

```powershell
python scripts/plot_single_task_auroc.py --inputs model_resnet.json model_vit.json model_swin.json --labels ResNet ViT Swin --output single_task_auroc.pdf
```

---

## 可选：`--doctor-json` 的格式模板

如果你还想叠加医生读片点，那么 `--doctor-json` 读取的不是列表，而是一个对象，且要求：

- 顶层 `record_type == "doctor_multi_task_roc_point_summary"`
- 必须包含 `plot_points` 列表

### 最小可用模板

```json
{
  "record_type": "doctor_multi_task_roc_point_summary",
  "plot_points": [
    {
      "task_label": "BM",
      "doctor_label": "doctor_1",
      "label": "BM - doctor_1",
      "roc_point": {
        "fpr": 0.144,
        "tpr": 0.728
      }
    },
    {
      "task_label": "BM",
      "doctor_label": "doctor_2",
      "label": "BM - doctor_2",
      "roc_point": {
        "fpr": 0.236,
        "tpr": 0.716
      }
    }
  ]
}
```

### 当前脚本实际会用到的字段

每个 `plot_points[i]` 至少建议有：

- `task_label`: 用于匹配 `--task-label`
- `doctor_label`: 用于筛选 `--doctor-labels`
- `roc_point.fpr`
- `roc_point.tpr`

### 完整示例

```json
{
  "record_type": "doctor_multi_task_roc_point_summary",
  "schema": "doctor_csv_multi_task_multi_doctor",
  "n_tasks": 1,
  "tasks": [],
  "plot_points": [
    {
      "task_label": "BM",
      "doctor_label": "doctor_1",
      "label": "BM - doctor_1",
      "input_csv": "BM.csv",
      "n_samples": 500,
      "metrics": {
        "accuracy": 0.792,
        "precision": 0.8349,
        "recall": 0.728,
        "specificity": 0.856,
        "fpr": 0.144,
        "f1": 0.7778
      },
      "roc_point": {
        "fpr": 0.144,
        "tpr": 0.728
      }
    }
  ]
}
```

---

## 一句话结论

如果你现在要把 **多个模型输出统一成可被 `plot_single_task_auroc.py` 读取的格式**：

- **最省事**：统一成 `roc_summary` 格式
- **最通用**：统一成 `true_label + prob_class_1` 的样本级格式

如果你愿意，下一步我可以继续帮你再补一个“多模型统一输出 JSON 的 Python 生成脚本模板”。
