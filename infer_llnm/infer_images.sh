#!/usr/bin/env bash
cd "$(dirname "$0")"

# ---------- 在这里改 ----------
# 训练好的权重文件（.pth）
MODEL_PATH=./your_model.pth
# 图像根目录；只用清单里的 directory 型 JSON 时可留空
IMAGE_DIR=./my_images
# 训练时的归一化参数（.pkl）；训练若开过归一化则填上，没有就留空
NORM_PKL=
# 清单文件（.json 或 .csv）；不用清单就留空，只扫 IMAGE_DIR
MANIFEST=
# 推理结果写到哪里（.csv）
OUTPUT_CSV=./inference_results.csv
# 每批多少张图
BATCH_SIZE=4
# cuda 用 GPU
DEVICE=cuda
# 留空的变量不会传给 infer_images.py
# ----------------------------

python infer_images.py \
  --model_path "$MODEL_PATH" \
  ${IMAGE_DIR:+--image_dir "$IMAGE_DIR"} \
  ${NORM_PKL:+--norm_params_file "$NORM_PKL"} \
  ${MANIFEST:+--manifest "$MANIFEST"} \
  --output_csv "$OUTPUT_CSV" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"
