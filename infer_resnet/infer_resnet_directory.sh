#!/usr/bin/env bash
# 在 infer_resnet 目录执行: bash infer_resnet_directory.sh
# 默认权重与数据相对「上一级」Tiger-Model 仓库根目录；可按需改成绝对路径

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PTH_PATH="$ROOT_DIR/modelsaved/weight.pth"
INPUT_DIR="$ROOT_DIR/dataset/test"
OUTPUT_CSV="$SCRIPT_DIR/output/predictions.csv"
NUM_CLASSES=2
BATCH_SIZE=4
DEVICE="cuda:0"

mkdir -p "$(dirname "$OUTPUT_CSV")"

python infer_resnet_directory.py \
        --pth_path "$PTH_PATH" \
        --input_dir "$INPUT_DIR" \
        --output_csv "$OUTPUT_CSV" \
        --num_classes $NUM_CLASSES \
        --batch_size $BATCH_SIZE \
        --device $DEVICE
