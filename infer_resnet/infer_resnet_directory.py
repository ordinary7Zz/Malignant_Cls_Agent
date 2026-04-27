#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对指定目录下的图像批量推理（ResNet-18，与 test_model.py / Resent 训练默认配置一致）。

预处理与验证集一致：CenterCrop(512) -> Resize(224) -> Normalize(0.5, ...)

用法示例:
  python infer_resnet_directory.py --pth_path ./modelsaved/MyModel/best.pth --input_dir ./my_images
  python infer_resnet_directory.py --pth_path ./modelsaved/x/best_model_weights.pth --input_dir ./imgs --num_classes 4 --output_csv ./preds.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: str, recursive: bool) -> List[str]:
    paths: List[str] = []
    root = os.path.abspath(root)
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                ext = os.path.splitext(name)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return paths


class ImageFolderInferenceDataset(Dataset):
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法读取图像: {path} ({e})") from e
        if self.transform:
            image = self.transform(image)
        return image, path


def build_model(num_classes: int, architecture: str) -> nn.Module:
    architecture = architecture.lower().strip()
    if architecture != "resnet":
        raise ValueError(f"当前仅支持 architecture=resnet，收到: {architecture}")
    model = models.resnet18(weights=None)
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(features, num_classes))
    return model


def load_weights(model: nn.Module, pth_path: str, device: torch.device) -> None:
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state, strict=False)


def default_eval_transform():
    return transforms.Compose(
        [
            transforms.CenterCrop(512),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_paths, inputs in tqdm(
        _batched_paths_and_tensors(loader), desc="推理", unit="batch"
    ):
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i, path in enumerate(batch_paths):
            row = {
                "path": path,
                "filename": os.path.basename(path),
                "pred_class": int(preds[i]),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = float(probs[i, c])
            rows.append(row)
    return pd.DataFrame(rows)


def _batched_paths_and_tensors(loader: DataLoader):
    """DataLoader yields (tensor_batch, path_list); path_batch is tuple from default_collate -> list of str per batch."""
    for tensor_batch, path_batch in loader:
        if isinstance(path_batch, str):
            paths = [path_batch]
        else:
            paths = list(path_batch)
        yield paths, tensor_batch


def parse_args():
    p = argparse.ArgumentParser(description="对目录内图像进行 ResNet 分类推理")
    p.add_argument("--pth_path", type=str, required=True, help="权重 .pth 路径（state_dict）")
    p.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    p.add_argument("--num_classes", type=int, default=2, help="类别数，需与训练时一致")
    p.add_argument("--architecture", type=str, default="resnet", choices=["resnet"], help="骨干，当前仅 resnet18")
    p.add_argument("--output_csv", type=str, default=None, help="预测结果 CSV；默认写到输入目录下 predictions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0 | cuda:1 ...（与 torch.device 一致）",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的图像",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.pth_path):
        raise FileNotFoundError(f"权重不存在: {args.pth_path}")
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"输入目录不存在: {args.input_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_paths = list_images(args.input_dir, args.recursive)
    if not image_paths:
        raise RuntimeError(f"目录下未找到支持的图像: {args.input_dir}")

    transform = default_eval_transform()
    dataset = ImageFolderInferenceDataset(image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.num_classes, args.architecture)
    load_weights(model, args.pth_path, device)
    model = model.to(device)

    df = run_inference(model, loader, device, args.num_classes)
    out_csv = args.output_csv or os.path.join(
        os.path.abspath(args.input_dir), "predictions.csv"
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"完成: {len(df)} 张图像 -> {out_csv}")


if __name__ == "__main__":
    main()
