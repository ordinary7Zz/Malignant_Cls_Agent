#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="python3"
OUTPUT_PATH="output/figures/LNMCN01_auroc.pdf"
DOCTOR_JSON=""
TITLE=""

declare -a INPUTS=()
declare -a LABELS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/plot_lnm_auroc_ubuntu.sh \
    --inputs lnm_model_a.json lnm_model_b.json lnm_model_c.json \
    --labels ThyroidAgent BestSingleModel AblationModel \
    [--doctor-json ./doctor_multi_task_metrics.json] \
    [--output output/figures/LNMCN01_auroc.pdf] \
    [--python python3] \
    [--title "LNM AUROC Comparison"]

Required:
  --inputs       LNM 任务的一个或多个 results JSON
  --labels       与输入 JSON 一一对应的标签

Optional:
  --doctor-json  医生读片点汇总 JSON；传入后会叠加 LNMCN01 医生点
  --output       输出路径，默认 output/figures/LNMCN01_auroc.pdf
  --python       Python 可执行文件，默认 python3
  --title        图标题；默认不显示

Example based on my_run.md:
  bash scripts/plot_lnm_auroc_ubuntu.sh \
    --inputs \
      /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/Malignant_Cls_Agent/output/test_dataset/LymphUs/results_20260517_160351.json \
      /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/Malignant_Cls_Agent/output/test_dataset/LymphUs/SingleModel/results_20260523_223045.json \
    --labels ThyroidAgent BestSingleModel
EOF
}

need_single_value() {
  local option_name="$1"
  local option_value="${2-}"
  if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
    echo "缺少 ${option_name} 的参数值。" >&2
    exit 1
  fi
}

collect_values() {
  local target_name="$1"
  shift
  declare -n target_ref="${target_name}"
  target_ref=()
  while (($#)) && [[ "$1" != --* ]]; do
    target_ref+=("$1")
    shift
  done
  COLLECTED_COUNT=$(( ${#target_ref[@]} + 1 ))
}

if (($# == 0)); then
  usage
  exit 1
fi

while (($#)); do
  case "$1" in
    --inputs)
      collect_values INPUTS "${@:2}"
      shift "${COLLECTED_COUNT}"
      ;;
    --labels)
      collect_values LABELS "${@:2}"
      shift "${COLLECTED_COUNT}"
      ;;
    --doctor-json)
      need_single_value "$1" "${2-}"
      DOCTOR_JSON="$2"
      shift 2
      ;;
    --output)
      need_single_value "$1" "${2-}"
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --python)
      need_single_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --title)
      need_single_value "$1" "${2-}"
      TITLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ((${#INPUTS[@]} == 0)); then
  echo "LNM 没有提供任何 JSON 输入。" >&2
  exit 1
fi
if ((${#LABELS[@]} == 0)); then
  echo "LNM 没有提供任何标签。" >&2
  exit 1
fi
if ((${#INPUTS[@]} != ${#LABELS[@]})); then
  echo "LNM 的 JSON 数量与标签数量不一致。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cmd=(
  "${PYTHON_BIN}" "scripts/plot_single_task_auroc.py"
  --inputs "${INPUTS[@]}"
  --labels "${LABELS[@]}"
  --task-label "LNMCN01"
  --output "${OUTPUT_PATH}"
)

if [[ -n "${DOCTOR_JSON}" ]]; then
  cmd+=(--doctor-json "${DOCTOR_JSON}")
fi
if [[ -n "${TITLE}" ]]; then
  cmd+=(--title "${TITLE}")
fi

"${cmd[@]}"
