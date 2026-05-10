"""
对图像目录进行 LLNM-Net 推理。

输入方式（``--manifest`` 与 ``--metadata_csv`` 二选一；皆无则用 ``--image_dir`` 扫目录）:

1) 仅 ``--image_dir``：递归扫描目录，多模态全部用命令行默认值。
2) ``--manifest`` / ``--metadata_csv`` 指向 **CSV**：每行一张图 + 可选列 report/age/sex/shape/echo（须配合 ``--image_dir``）。
3) ``--manifest`` 指向 **JSON**，两种形态（可用 ``type`` 显式声明，否则按 ``items`` / ``image_dir`` 自动推断）:

   - **directory**：整目录推理，全部默认数值。示例::

        {
          "type": "directory",
          "image_dir": "D:/data/batch1",
          "recursive": true,
          "default_report": "",
          "default_age": 50,
          "default_sex": 1,
          "default_shape_echo": [0, 0]
        }

   - **images**：``image_dir`` + 每条 ``image``（相对路径）；可选 ``report`` / ``age`` /
     ``sex`` / ``shape`` / ``echo`` 等。``items`` 里也可直接写相对路径字符串。示例::

        {
          "type": "images",
          "image_dir": "D:/data/root",
          "items": [
            {"image": "case/a.jpg", "report": "甲状腺结节", "age": 45, "sex": 1},
            "case/b.jpg"
          ]
        }

4) ``--metadata_csv``：与 manifest 为 CSV 时等价，保留兼容。

若训练时使用了 prepare_data 的归一化，请提供 train_norm_params.pkl。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from Malignant_Cls_Agent.infer_llnm.models.modeling_LLNM_Net import CONFIGS, LLNM_Net

tk_lim = 300  # 与 configs.py 中 rr_len 一致


def parse_device_arg(s: str) -> str:
    """接受 cpu、cuda 或 cuda:N（指定 GPU 编号）。"""
    s = s.strip()
    if s in ("cpu", "cuda"):
        return s
    if s.startswith("cuda:") and s[5:].isdigit():
        return s
    raise argparse.ArgumentTypeError(
        f"invalid device {s!r}: use 'cpu', 'cuda', or 'cuda:N' (e.g. cuda:0)"
    )


def _extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"不支持的权重格式: {type(checkpoint)!r}")
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    return checkpoint


def _load_checkpoint_state_dict(weight_path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    return _extract_state_dict(checkpoint)


def _infer_num_classes_from_state_dict(
    pretrained_weights: dict[str, torch.Tensor],
) -> int | None:
    head_weight = pretrained_weights.get("head.weight")
    if isinstance(head_weight, torch.Tensor) and head_weight.ndim == 2:
        return int(head_weight.shape[0])
    head_bias = pretrained_weights.get("head.bias")
    if isinstance(head_bias, torch.Tensor) and head_bias.ndim == 1:
        return int(head_bias.shape[0])
    return None


def load_weights(
    model: torch.nn.Module,
    weight_path: str,
    pretrained_weights: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    pretrained_weights = pretrained_weights or _load_checkpoint_state_dict(weight_path)
    model_weights = model.state_dict()
    loadable_weights = {}
    skipped_shape_mismatch = []
    for key, value in pretrained_weights.items():
        if key not in model_weights:
            continue
        if model_weights[key].shape != value.shape:
            skipped_shape_mismatch.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_weights[key].shape)}"
            )
            continue
        loadable_weights[key] = value
    model_weights.update(loadable_weights)
    model.load_state_dict(model_weights)
    if skipped_shape_mismatch:
        print("跳过尺寸不匹配的权重:")
        for item in skipped_shape_mismatch:
            print(f"  - {item}")
    print(f"已加载权重: {weight_path}")
    return model


def _precompute_rr(
    tokenizer: BertTokenizer,
    bert_model: BertModel,
    default_report: str,
) -> torch.Tensor:
    with torch.no_grad():
        input_ids = tokenizer.encode(
            default_report, add_special_tokens=True, return_tensors="pt"
        )
        outputs = bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        padding_length = tk_lim - last_hidden_state.shape[1]
        if padding_length > 0:
            padding_token = tokenizer.pad_token_id
            padding_tensor = torch.full(
                (1, padding_length, last_hidden_state.shape[2]), padding_token
            )
            padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
        else:
            padded_outputs = last_hidden_state
        rr_vector = padded_outputs[:, :tk_lim, :]
    return rr_vector.squeeze(0).float().contiguous()


def _load_norm_params(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        norm_params = pickle.load(f)
    req = [
        "age_mean",
        "age_std",
        "shape_mean",
        "shape_std",
        "echo_mean",
        "echo_std",
    ]
    missing = [k for k in req if k not in norm_params]
    if missing:
        raise ValueError(f"归一化文件缺少键: {missing}")
    print(f"已加载归一化参数: {path}")
    return norm_params


def _norm_bics_bts_tensors(
    age: float,
    sex: float,
    shape_v: float,
    echo_v: float,
    norm_params: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    age, sex = float(age), float(sex)
    shape_v, echo_v = float(shape_v), float(echo_v)
    if norm_params is not None:
        age = (age - norm_params["age_mean"]) / norm_params["age_std"]
        shape_v = (shape_v - norm_params["shape_mean"]) / norm_params["shape_std"]
        echo_v = (echo_v - norm_params["echo_mean"]) / norm_params["echo_std"]
    demo = torch.tensor([age, sex], dtype=torch.float32)
    img_fea = torch.tensor([shape_v, echo_v], dtype=torch.float32)
    return demo, img_fea


class ImageDirInferenceDataset(Dataset):
    """仅遍历图像目录；多模态侧为常数（与训练时维度一致）。"""

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        image_dir: str,
        transform,
        default_report: str = "",
        default_age: float = 50.0,
        default_sex: float = 1.0,
        default_shape_echo: tuple[float, float] | None = None,
        norm_params_file: str | None = None,
        recursive: bool = True,
    ):
        self.root = Path(image_dir).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"图像目录不存在或不是文件夹: {self.root}")

        self.paths: list[Path] = []
        it = self.root.rglob("*") if recursive else self.root.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXT:
                self.paths.append(p)

        self.paths.sort(key=lambda x: str(x))
        if not self.paths:
            raise FileNotFoundError(f"目录下未找到支持的图像: {self.root}")

        self.transform = transform

        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.bert_model = BertModel.from_pretrained("bert-base-chinese")
            self.bert_model.eval()
        except Exception as e:
            print(f"加载 BERT 失败: {e}", file=sys.stderr)
            print("可设置环境变量 HF_ENDPOINT 或使用本地模型路径。", file=sys.stderr)
            raise

        norm_params = _load_norm_params(norm_params_file)

        if default_shape_echo is None:
            default_shape_echo = (0.0, 0.0)

        self._demo, self._img_fea = _norm_bics_bts_tensors(
            default_age,
            default_sex,
            default_shape_echo[0],
            default_shape_echo[1],
            norm_params,
        )
        self._rr = _precompute_rr(self.tokenizer, self.bert_model, default_report)

        print(f"共 {len(self.paths)} 张图像，根目录: {self.root}")
        print(
            f"默认值(归一化后若提供 norm 文件): report 长度字符={len(default_report)}, "
            f"age/sex 张量={self._demo.tolist()}, shape/echo={self._img_fea.tolist()}"
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法读取 {path}: {e}，使用黑图占位")
            img = Image.new("RGB", (224, 224), color="black")
        if self.transform:
            img = self.transform(img)
        rel = str(path.relative_to(self.root)).replace("\\", "/")
        return img, rel, self._rr.clone(), self._demo.clone(), self._img_fea.clone()


def _csv_cell_str(row: pd.Series, col: str | None, default: str) -> str:
    if col is None:
        return default
    v = row[col]
    if pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def build_records_from_csv(metadata_csv: str, image_dir: Path) -> list[dict]:
    if not os.path.isfile(metadata_csv):
        raise FileNotFoundError(f"元数据 CSV 不存在: {metadata_csv}")
    root = image_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"图像目录不存在或不是文件夹: {root}")

    df = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    key_col = col("image", "relative_path", "rel_path", "path")
    if key_col is None:
        raise ValueError("CSV 必须包含列之一: image, relative_path, rel_path, path")
    c_report = col("report", "text", "超声报告")
    c_age = col("age", "年龄")
    c_sex = col("sex", "gender", "性别")
    c_shape = col("shape", "形态", "shape_feature")
    c_echo = col("echo", "回声", "echo_feature")

    exts = ImageDirInferenceDataset.IMAGE_EXT
    records: list[dict] = []
    missing_files: list[str] = []
    for _, row in df.iterrows():
        rel = str(row[key_col]).strip().replace("\\", "/")
        if not rel or rel.lower() == "nan":
            continue
        full = (root / rel).resolve()
        if not full.is_file():
            missing_files.append(rel)
            continue
        if full.suffix.lower() not in exts:
            continue
        rec: dict = {"path": full, "out_key": rel.replace("\\", "/")}
        if c_report is not None:
            s = _csv_cell_str(row, c_report, "")
            if s:
                rec["report"] = s
        if c_age is not None and not (
            pd.isna(row[c_age]) or str(row[c_age]).strip() == ""
        ):
            rec["age"] = float(row[c_age])
        if c_sex is not None and not (
            pd.isna(row[c_sex]) or str(row[c_sex]).strip() == ""
        ):
            rec["sex"] = float(row[c_sex])
        if c_shape is not None and not (
            pd.isna(row[c_shape]) or str(row[c_shape]).strip() == ""
        ):
            rec["shape"] = float(row[c_shape])
        if c_echo is not None and not (
            pd.isna(row[c_echo]) or str(row[c_echo]).strip() == ""
        ):
            rec["echo"] = float(row[c_echo])
        records.append(rec)

    if missing_files:
        print(f"警告: CSV 中有 {len(missing_files)} 条路径在磁盘上不存在（已跳过）。")
        if len(missing_files) <= 10:
            for m in missing_files:
                print(f"  缺失: {m}")
    if not records:
        raise FileNotFoundError(
            f"CSV 中无有效图像行（请检查路径列是否相对 image_dir）: {metadata_csv}"
        )
    print(f"自 CSV 读取 {len(records)} 张图: {metadata_csv}")
    return records


def _json_get_first(d: dict, keys: tuple[str, ...]) -> object | None:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None


def _resolve_item_image(image_root: Path, item: dict) -> tuple[Path, str]:
    """``image_root / item['image']``；``image`` 须为相对路径字符串。"""
    v = item.get("image")
    if v is None or str(v).strip() == "":
        raise ValueError(
            f'items 每项须包含非空 "image" 字段（相对 image_dir）: {item!r}'
        )
    rel_s = str(v).strip().replace("\\", "/")
    full = (image_root / rel_s).resolve()
    return full, rel_s


def build_records_from_json_items(
    data: dict, cli_image_dir: Path | None
) -> list[dict]:
    raw_dir = data.get("image_dir")
    if raw_dir is not None and str(raw_dir).strip():
        image_root = Path(str(raw_dir).strip()).expanduser().resolve()
    elif cli_image_dir is not None:
        image_root = cli_image_dir.resolve()
    else:
        image_root = None
    if image_root is None:
        raise ValueError(
            '逐张 JSON 须在根上提供 "image_dir"，或在命令行提供 --image_dir'
        )

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError('JSON（逐张模式）必须包含数组字段 "items"')
    if len(items) == 0:
        raise ValueError('"items" 数组不能为空')

    exts = ImageDirInferenceDataset.IMAGE_EXT
    records: list[dict] = []
    skipped: list[str] = []
    for it in items:
        if isinstance(it, str):
            it = {"image": it.strip()}
        if not isinstance(it, dict):
            raise ValueError(f"items 中每项须为对象或字符串，收到: {type(it)}")
        path, out_key = _resolve_item_image(image_root, it)
        if not path.is_file():
            skipped.append(out_key)
            continue
        if path.suffix.lower() not in exts:
            continue
        rec: dict = {"path": path, "out_key": out_key}
        for fld, jkeys in (
            ("report", ("report", "text", "超声报告")),
            ("age", ("age", "年龄")),
            ("sex", ("sex", "gender", "性别")),
            ("shape", ("shape", "形态", "shape_feature")),
            ("echo", ("echo", "回声", "echo_feature")),
        ):
            v = _json_get_first(it, jkeys)
            if v is not None and str(v).strip() != "":
                rec[fld] = str(v).strip() if fld == "report" else float(v)
        records.append(rec)

    if skipped:
        print(f"警告: JSON 中有 {len(skipped)} 条路径在磁盘上不存在（已跳过）。")
        if len(skipped) <= 10:
            for s in skipped:
                print(f"  缺失: {s}")
    if not records:
        raise FileNotFoundError("JSON items 中无有效图像，请检查路径与 image_dir")
    print(f"自 JSON 列表读取 {len(records)} 张图")
    return records


def _directory_payload_from_dict(data: dict) -> dict:
    raw = _json_get_first(data, ("image_dir", "root", "image_root", "dir"))
    if raw is None:
        raise ValueError('directory 型 manifest 须包含 "image_dir" 或 "root" 等字段')
    image_dir = str(raw).strip()
    recursive = bool(data.get("recursive", True))
    overrides = {
        k: data[k]
        for k in (
            "default_report",
            "default_age",
            "default_sex",
            "default_shape_echo",
            "norm_params_file",
        )
        if k in data
    }
    if "default_shape_echo" in overrides:
        dse = overrides["default_shape_echo"]
        if isinstance(dse, str):
            overrides["default_shape_echo"] = [
                float(x.strip()) for x in dse.split(",")
            ]
        elif isinstance(dse, (list, tuple)) and len(dse) >= 2:
            overrides["default_shape_echo"] = [float(dse[0]), float(dse[1])]
    return {
        "image_dir": image_dir,
        "recursive": recursive,
        "overrides": overrides,
    }


def load_manifest_json(
    path: str, cli_image_dir: str | None
) -> tuple[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("manifest JSON 顶层须为对象")

    explicit = data.get("type") or data.get("manifest_type")
    et = str(explicit).strip().lower() if explicit is not None else None
    items = data.get("items")
    nonempty_items = isinstance(items, list) and len(items) > 0

    def _record_overrides() -> dict:
        return {
            k: data[k]
            for k in (
                "default_report",
                "default_age",
                "default_sex",
                "default_shape_echo",
                "norm_params_file",
            )
            if k in data
        }

    if et in ("images", "list", "per_image", "items"):
        cli_p = Path(cli_image_dir).resolve() if cli_image_dir else None
        records = build_records_from_json_items(data, cli_p)
        ov = _record_overrides()
        if "default_shape_echo" in ov:
            dse = ov["default_shape_echo"]
            if isinstance(dse, str):
                ov["default_shape_echo"] = [float(x.strip()) for x in dse.split(",")]
            elif isinstance(dse, (list, tuple)) and len(dse) >= 2:
                ov["default_shape_echo"] = [float(dse[0]), float(dse[1])]
        return "records", {"records": records, "overrides": ov}

    if et in ("directory", "dir", "folder"):
        return "directory", _directory_payload_from_dict(data)

    if nonempty_items:
        cli_p = Path(cli_image_dir).resolve() if cli_image_dir else None
        records = build_records_from_json_items(data, cli_p)
        ov = _record_overrides()
        if "default_shape_echo" in ov:
            dse = ov["default_shape_echo"]
            if isinstance(dse, str):
                ov["default_shape_echo"] = [float(x.strip()) for x in dse.split(",")]
            elif isinstance(dse, (list, tuple)) and len(dse) >= 2:
                ov["default_shape_echo"] = [float(dse[0]), float(dse[1])]
        return "records", {"records": records, "overrides": ov}

    if _json_get_first(data, ("image_dir", "root", "image_root", "dir")) is not None:
        return "directory", _directory_payload_from_dict(data)

    raise ValueError(
        "无法解析 manifest JSON：请设置 type 为 directory 或 images，"
        "或提供 image_dir（整目录）或非空 items（逐张）"
    )


class ImageRecordsInferenceDataset(Dataset):
    """由规范化记录列表驱动：每条含绝对 path、输出键及可选多模态字段。"""

    def __init__(
        self,
        records: list[dict],
        transform,
        default_report: str = "",
        default_age: float = 50.0,
        default_sex: float = 1.0,
        default_shape_echo: tuple[float, float] | None = None,
        norm_params_file: str | None = None,
    ):
        self.records = records
        self.transform = transform
        self.default_report = default_report or ""
        self.default_age = float(default_age)
        self.default_sex = float(default_sex)
        if default_shape_echo is None:
            default_shape_echo = (0.0, 0.0)
        self.default_shape = float(default_shape_echo[0])
        self.default_echo = float(default_shape_echo[1])

        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.bert_model = BertModel.from_pretrained("bert-base-chinese")
            self.bert_model.eval()
        except Exception as e:
            print(f"加载 BERT 失败: {e}", file=sys.stderr)
            raise

        self.norm_params = _load_norm_params(norm_params_file)
        self._rr_cache: dict[str, torch.Tensor] = {}

    def _rr_for_report(self, report: str) -> torch.Tensor:
        if report not in self._rr_cache:
            self._rr_cache[report] = _precompute_rr(
                self.tokenizer, self.bert_model, report
            )
        return self._rr_cache[report].clone()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path: Path = rec["path"]
        out_key: str = rec["out_key"]
        report = rec.get("report", self.default_report) or self.default_report
        age = float(rec["age"]) if "age" in rec else self.default_age
        sex = float(rec["sex"]) if "sex" in rec else self.default_sex
        shape_v = float(rec["shape"]) if "shape" in rec else self.default_shape
        echo_v = float(rec["echo"]) if "echo" in rec else self.default_echo

        demo, img_fea = _norm_bics_bts_tensors(age, sex, shape_v, echo_v, self.norm_params)
        rr = self._rr_for_report(report)

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法读取 {path}: {e}，使用黑图占位")
            img = Image.new("RGB", (224, 224), color="black")
        if self.transform:
            img = self.transform(img)
        return img, out_key.replace("\\", "/"), rr, demo, img_fea


class ImageMetadataInferenceDataset(ImageRecordsInferenceDataset):
    """兼容旧名：由 CSV + image_dir 构建。"""

    def __init__(
        self,
        image_dir: str,
        metadata_csv: str,
        transform,
        default_report: str = "",
        default_age: float = 50.0,
        default_sex: float = 1.0,
        default_shape_echo: tuple[float, float] | None = None,
        norm_params_file: str | None = None,
    ):
        records = build_records_from_csv(metadata_csv, Path(image_dir))
        super().__init__(
            records,
            transform,
            default_report=default_report,
            default_age=default_age,
            default_sex=default_sex,
            default_shape_echo=default_shape_echo,
            norm_params_file=norm_params_file,
        )


def collate_fn(batch):
    imgs, rels, rrs, demos, img_feas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    rrs = torch.stack(rrs, dim=0)
    demos = torch.stack(demos, dim=0)
    img_feas = torch.stack(img_feas, dim=0)
    return imgs, list(rels), rrs, demos, img_feas


@torch.inference_mode()
def run_inference(
    model_path: str,
    dataset: Dataset,
    batch_size: int,
    num_classes: int | None,
    device: str,
) -> tuple[list[str], np.ndarray]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        device = "cpu"
    device_t = torch.device(device)

    pretrained_weights = _load_checkpoint_state_dict(model_path)
    inferred_num_classes = _infer_num_classes_from_state_dict(pretrained_weights)
    if num_classes is None:
        if inferred_num_classes is None:
            raise ValueError("无法从权重中推断 num_classes，请显式传入 --num_classes")
        num_classes = inferred_num_classes
        print(f"从权重自动识别 num_classes={num_classes}")
    elif inferred_num_classes is not None and inferred_num_classes != num_classes:
        print(
            f"警告: 传入 num_classes={num_classes}，但权重分类头输出为 {inferred_num_classes}；将按传入值构建模型"
        )

    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    model = load_weights(model, model_path, pretrained_weights=pretrained_weights)
    model = model.to(device_t)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_t.type == "cuda"),
        collate_fn=collate_fn,
    )

    all_rel: list[str] = []
    all_probs: list[np.ndarray] = []

    for imgs, rels, rr, demo, img_fea in tqdm(loader, desc="推理"):
        imgs = imgs.to(device_t)
        rr = rr.view(-1, tk_lim, rr.shape[2]).to(device_t).float()
        demo = demo.view(-1, 1, demo.shape[1]).to(device_t).float()
        img_fea = img_fea.view(-1, img_fea.shape[1], 1).to(device_t).float()
        sex = demo[:, :, 1].view(-1, 1, 1).to(device_t).float()
        age = demo[:, :, 0].view(-1, 1, 1).to(device_t).float()

        logits = model(imgs, rr, img_fea, sex, age)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_rel.extend(rels)
        all_probs.append(probs)

    return all_rel, np.concatenate(all_probs, axis=0)


def save_csv(
    paths: list[str],
    probs: np.ndarray,
    out_path: str,
    threshold: float,
) -> None:
    header = ["relative_path"]
    if probs.shape[1] == 1:
        header.extend(["prob_class0", "prob_class1"])
    else:
        header.extend([f"prob_class{i}" for i in range(probs.shape[1])])
    header.append("pred_class")

    lines = [",".join(header)]
    for i, p in enumerate(paths):
        if probs.shape[1] == 1:
            prob1 = float(probs[i, 0])
            prob0 = 1.0 - prob1
            pred = int(prob1 >= threshold)
            row = [p, f"{prob0:.6f}", f"{prob1:.6f}", str(pred)]
        else:
            row = [p] + [f"{probs[i, j]:.6f}" for j in range(probs.shape[1])]
            pred = int(np.argmax(probs[i]))
            if probs.shape[1] == 2:
                pred = int(probs[i, 1] >= threshold)
            row.append(str(pred))
        lines.append(",".join(row))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"结果已写入: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLNM-Net 图像目录推理（多模态默认值）")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件（键名与下列参数一致，不含 -- 前缀）",
    )
    p.add_argument("--model_path", type=str, default=None, help="模型 .pth 路径")
    p.add_argument("--image_dir", type=str, default=None, help="待推理图像根目录")
    p.add_argument(
        "--norm_params_file",
        type=str,
        default=None,
        help="训练阶段生成的 train_norm_params.pkl（训练若启用归一化则强烈建议提供）",
    )
    p.add_argument("--output_csv", type=str, default="inference_results.csv", help="输出 CSV")
    p.add_argument("--default_report", type=str, default="", help="默认超声报告文本")
    p.add_argument("--default_age", type=float, default=50.0, help="默认年龄（归一化前）")
    p.add_argument("--default_sex", type=float, default=1.0, help="默认性别 0=女 1=男")
    p.add_argument(
        "--default_shape_echo",
        type=str,
        default="0,0",
        help='默认形态与回声特征（归一化前），格式 "shape,echo"',
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=None, help="类别数；留空时从权重分类头自动识别")
    p.add_argument(
        "--device",
        type=parse_device_arg,
        default="cuda",
        help="推理设备：cpu、cuda（默认 GPU）或 cuda:N（如 cuda:0）",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="二分类时若用 argmax 与 prob_class1>=threshold 不一致，CSV 中 pred_class 对二类采用此阈值判定正类",
    )
    p.add_argument(
        "--no_recursive",
        action="store_true",
        help="仅扫描 image_dir 下第一层文件，不递归子目录",
    )
    p.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="同 --manifest 为 .csv 时的用法（保留兼容）；不要与 --manifest 同时使用",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="输入清单：.json（directory 整目录 / images 为 image_dir+items[].image）或 .csv；见文件顶部说明",
    )
    return p


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    defaults: dict = {}
    if pre_args.config:
        with open(pre_args.config, encoding="utf-8") as f:
            defaults = json.load(f)
        if not isinstance(defaults, dict):
            raise ValueError("配置文件顶层必须是 JSON 对象")

    parser = build_parser()
    parser.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = parser.parse_args()

    if not args.model_path:
        parser.error("必须提供 model_path（可在 JSON 配置中给出）")
    if not os.path.isfile(args.model_path):
        parser.error(f"模型文件不存在: {args.model_path}")

    mp = args.manifest or args.metadata_csv
    if args.manifest and args.metadata_csv:
        parser.error("不要同时使用 --manifest 与 --metadata_csv")
    if mp and Path(mp).suffix.lower() == ".csv" and not args.image_dir:
        parser.error("CSV 清单需要 --image_dir 作为图像相对路径的根目录")
    if not mp and not args.image_dir:
        parser.error("请提供 --image_dir，或使用 --manifest / --metadata_csv 指向清单文件")
    return args


def _effective_inference_params(
    args: argparse.Namespace, overrides: dict | None
) -> tuple[str, float, float, tuple[float, float], str | None]:
    """命令行参数与 manifest 内可选默认字段合并。"""
    dr = args.default_report
    da = args.default_age
    ds = args.default_sex
    se = [float(x.strip()) for x in args.default_shape_echo.split(",")]
    nf: str | None = args.norm_params_file
    if overrides:
        if "default_report" in overrides and overrides["default_report"] is not None:
            dr = str(overrides["default_report"])
        if "default_age" in overrides and overrides["default_age"] is not None:
            da = float(overrides["default_age"])
        if "default_sex" in overrides and overrides["default_sex"] is not None:
            ds = float(overrides["default_sex"])
        if "default_shape_echo" in overrides and overrides["default_shape_echo"] is not None:
            v = overrides["default_shape_echo"]
            if isinstance(v, str):
                se = [float(x.strip()) for x in v.split(",")]
            elif isinstance(v, (list, tuple)):
                se = [float(v[0]), float(v[1])]
        if overrides.get("norm_params_file"):
            nf = str(overrides["norm_params_file"])
    if len(se) != 2:
        raise SystemExit("default_shape_echo 必须为两个数值（shape, echo）")
    return dr, da, ds, (se[0], se[1]), nf


def main() -> None:
    args = parse_args()
    mp = args.manifest or args.metadata_csv

    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    if mp:
        suf = Path(mp).suffix.lower()
        if suf == ".csv":
            dr, da, ds, shape_tuple, nf = _effective_inference_params(args, None)
            records = build_records_from_csv(mp, Path(args.image_dir))
            dataset = ImageRecordsInferenceDataset(
                records,
                data_transforms,
                default_report=dr,
                default_age=da,
                default_sex=ds,
                default_shape_echo=shape_tuple,
                norm_params_file=nf,
            )
        elif suf == ".json":
            kind, payload = load_manifest_json(mp, args.image_dir)
            if kind == "directory":
                d_ov = payload.get("overrides") or {}
                dr, da, ds, shape_tuple, nf = _effective_inference_params(args, d_ov)
                dataset = ImageDirInferenceDataset(
                    image_dir=payload["image_dir"],
                    transform=data_transforms,
                    default_report=dr,
                    default_age=da,
                    default_sex=ds,
                    default_shape_echo=shape_tuple,
                    norm_params_file=nf,
                    recursive=payload["recursive"],
                )
            else:
                rov = payload.get("overrides") or {}
                dr, da, ds, shape_tuple, nf = _effective_inference_params(args, rov)
                dataset = ImageRecordsInferenceDataset(
                    payload["records"],
                    data_transforms,
                    default_report=dr,
                    default_age=da,
                    default_sex=ds,
                    default_shape_echo=shape_tuple,
                    norm_params_file=nf,
                )
        else:
            raise SystemExit(f"不支持的清单扩展名（仅 .json / .csv）: {mp}")
    else:
        dr, da, ds, shape_tuple, nf = _effective_inference_params(args, None)
        dataset = ImageDirInferenceDataset(
            image_dir=args.image_dir,
            transform=data_transforms,
            default_report=dr,
            default_age=da,
            default_sex=ds,
            default_shape_echo=shape_tuple,
            norm_params_file=nf,
            recursive=not args.no_recursive,
        )

    rels, probs = run_inference(
        model_path=args.model_path,
        dataset=dataset,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        device=args.device,
    )
    save_csv(rels, probs, args.output_csv, args.threshold)


if __name__ == "__main__":
    main()
