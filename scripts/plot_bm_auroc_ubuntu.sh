#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="python3"
OUTPUT_PATH="output/figures/BM_auroc.pdf"
DOCTOR_JSON=""
TITLE=""

declare -a INPUTS=()
declare -a LABELS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/plot_bm_auroc_ubuntu.sh \
    --inputs bm_model_a.json bm_model_b.json bm_model_c.json \
    --labels ThyroidAgent BestSingleModel AblationModel \
    [--doctor-json ./doctor_multi_task_metrics.json] \
    [--output output/figures/BM_auroc.pdf] \
    [--python python3] \
    [--title "BM (Benign/Malignant) AUROC Comparison"]

Required:
  --inputs       BM 任务的一个或多个 results JSON
  --labels       与输入 JSON 一一对应的标签

Optional:
  --doctor-json  医生读片点汇总 JSON；传入后会叠加 BM 医生点
  --output       输出路径，默认 output/figures/BM_auroc.pdf
  --python       Python 可执行文件，默认 python3
  --title        图标题；默认不显示

Example based on my_run.md:
  bash scripts/plot_bm_auroc_ubuntu.sh \
    --inputs \
      /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/output/500_TestData_Malignancy_Cls/no_llm/results_20260503_201231.json \
      /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/output/500_TestData_Malignancy_Cls/no_llm_single_model/results_20260522_215423.json \
    --labels ThyroidAgent BestSingleModel \
    --doctor-json ./doctor_multi_task_metrics.json
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
  echo "BM 没有提供任何 JSON 输入。" >&2
  exit 1
fi
if ((${#LABELS[@]} == 0)); then
  echo "BM 没有提供任何标签。" >&2
  exit 1
fi
if ((${#INPUTS[@]} != ${#LABELS[@]})); then
  echo "BM 的 JSON 数量与标签数量不一致。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cmd=(
  "${PYTHON_BIN}" "scripts/plot_single_task_auroc.py"
  --inputs "${INPUTS[@]}"
  --labels "${LABELS[@]}"
  --task-label "BM"
  --output "${OUTPUT_PATH}"
)

if [[ -n "${DOCTOR_JSON}" ]]; then
  cmd+=(--doctor-json "${DOCTOR_JSON}")
fi
if [[ -n "${TITLE}" ]]; then
  cmd+=(--title "${TITLE}")
fi

"${cmd[@]}"
