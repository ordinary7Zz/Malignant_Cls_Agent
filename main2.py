"""
甲状腺分类模型演示脚本（扩展版）

在 DINO-UNet 与 AutoGluon-PyRadiomics 流程基础上，可选地增加：

- 淋巴结转移二分类（LLNM-Net，逻辑同 ``infer_llnm/infer_images.py``）
- 转移病理亚型二分类（ResNet，逻辑同 ``infer_resnet/infer_resnet_directory.py``）

二分类推理由 ``scripts/auxiliary_binary_inference.py`` 提供。
"""

import os
import sys
import argparse
import numpy as np
import yaml
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_registry import ModelRegistry
from models.dino_unet_model import DINOUNetModel
from models.autogluon_radiomics_model import AutoGluonRadiomicsModel
from models.base_model import ModelOutput
from agent.Malignant_Cls_Agent import (
    LLMClassificationAgent,
    _average_class_probabilities,
    _winning_class_from_avg_probs,
)
from utils.image_processor import ImageProcessor
from calibration.runtime import (
    load_calibration_map_from_config,
    maybe_apply_calibration_map,
)
from scripts.auxiliary_binary_inference import (
    run_llnm_binary_for_image_paths,
    run_resnet_binary_for_image_paths,
)


def load_config(config_path: str = "config/config2.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_image_files(directory: str) -> List[Path]:
    """获取目录中所有图像文件"""
    directory = Path(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    image_files = []
    for ext in image_extensions:
        # 使用 glob 的不区分大小写匹配，避免重复
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))

    # 去重：转换为绝对路径的集合，然后排序
    unique_files = sorted(set(f.resolve() for f in image_files))
    return unique_files


def find_corresponding_mask(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """在掩码目录中查找与图像文件名对应的掩码文件"""
    # 尝试相同的扩展名
    mask_path = mask_dir / image_path.name
    if mask_path.exists():
        return mask_path

    # 尝试不同的扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for ext in image_extensions:
        mask_path = mask_dir / f"{image_path.stem}{ext}"
        if mask_path.exists():
            return mask_path

    return None


def infer_label_path_by_output_dir(output_dir: str) -> Optional[Path]:
    """按 output_dir 名称推断标签文件路径。"""
    out_name = Path(output_dir).name
    project_root = Path(__file__).resolve().parent

    candidates: list[Path] = []
    if "TN3K" in out_name:
        candidates = [
            project_root / "tn3k_test_label.json",
            project_root / "TN3K_test_label.json",
        ]
    elif "TN5K" in out_name:
        candidates = [project_root / "TN5K_test_label.json"]
    elif "ThyroidXL" in out_name:
        candidates = [project_root / "ThyroidXL_test_label.json"]
    else:
        candidates = [project_root / "test_label.json"]

    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_label_path(paths_config: dict) -> tuple[Optional[Path], str]:
    """
    解析标签文件路径。
    返回: (label_path_or_none, source)
      - source: "config" | "auto" | "none"
    """
    data_cfg = paths_config.get("data", {}) if isinstance(paths_config, dict) else {}
    configured_label = data_cfg.get("label_file", None)
    project_root = Path(__file__).resolve().parent

    if configured_label is not None:
        configured_label_str = str(configured_label).strip()
        if configured_label_str and configured_label_str.lower() != "null":
            configured_path = Path(configured_label_str)
            if not configured_path.is_absolute():
                configured_path = project_root / configured_path
            if configured_path.exists():
                return configured_path, "config"
            return None, "config"

    output_dir = str(paths_config.get("output", {}).get("output_dir", "output"))
    auto_path = infer_label_path_by_output_dir(output_dir)
    if auto_path is not None and auto_path.exists():
        return auto_path, "auto"
    return None, "none"


def _to_model_list(section: Any) -> List[dict]:
    """把配置段统一成模型配置列表。"""
    if section is None:
        return []
    if isinstance(section, list):
        return [x for x in section if isinstance(x, dict)]
    if isinstance(section, dict):
        nested = section.get("models")
        if isinstance(nested, list):
            return [x for x in nested if isinstance(x, dict)]
        return [section]
    return []


def _filter_enabled_models(model_list: List[dict]) -> List[dict]:
    """统一按 enable 过滤模型列表；未显式配置 enable 视为启用。"""
    return [cfg for cfg in model_list if bool(cfg.get("enable", True))]


def resolve_pipeline_model_configs(paths_config: dict) -> dict[str, List[dict]]:
    """
    统一解析四类模型配置，均作为二分类候选模型参与决策。

    仅支持如下结构：
      models:
        dino_unet: [ ... ]
        autogluon: [ ... ]
        llnm: [ ... ]
        pathology_subtype_resnet: [ ... ]
    """
    if not isinstance(paths_config, dict):
        raise ValueError("配置格式错误：根配置必须是字典")

    models_root = paths_config.get("models")
    if not isinstance(models_root, dict):
        raise ValueError("配置缺失：请在 config 中提供 models 节")

    out: dict[str, List[dict]] = {
        "dino_unet": [],
        "autogluon": [],
        "llnm": [],
        "pathology_subtype_resnet": [],
    }

    out["dino_unet"] = _filter_enabled_models(_to_model_list(models_root.get("dino_unet")))
    out["autogluon"] = _filter_enabled_models(_to_model_list(models_root.get("autogluon")))
    out["llnm"] = _filter_enabled_models(_to_model_list(models_root.get("llnm")))
    out["pathology_subtype_resnet"] = _filter_enabled_models(
        _to_model_list(models_root.get("pathology_subtype_resnet"))
    )

    return out


def _safe_normalize_binary_probs(prob_0: float, prob_1: float) -> tuple[float, float]:
    """规范化二分类概率，避免异常值影响后续融合。"""
    p0 = max(0.0, min(1.0, float(prob_0)))
    p1 = max(0.0, min(1.0, float(prob_1)))
    s = p0 + p1
    if s <= 0.0:
        return 0.5, 0.5
    return p0 / s, p1 / s


def _binary_entropy(prob_0: float, prob_1: float) -> float:
    eps = 1e-12
    p0 = max(eps, min(1.0, float(prob_0)))
    p1 = max(eps, min(1.0, float(prob_1)))
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def _build_binary_model_output(
    binary_record: dict,
    model_name: str,
    task_name: str,
    test_set_performance: Optional[Dict[str, Any]] = None,
) -> ModelOutput:
    """将二分类输出转换为统一 ModelOutput，供 Agent 统一决策。"""
    prob_0, prob_1 = _safe_normalize_binary_probs(
        binary_record.get("prob_class_0", 0.5),
        binary_record.get("prob_class_1", 0.5),
    )
    predictions = {
        "0": float(prob_0),
        "1": float(prob_1),
    }
    top_class = "1" if prob_1 >= prob_0 else "0"
    top_confidence = float(max(prob_0, prob_1))

    metadata: Dict[str, Any] = {
        "source": "unified_binary_classification",
        "task": task_name,
        "positive_class_index": 1,
        "classification_uncertainty": {
            "top_confidence_raw": top_confidence,
            "entropy": _binary_entropy(prob_0, prob_1),
            "margin_top2": float(abs(prob_1 - prob_0)),
        },
    }
    if test_set_performance:
        metadata["test_set_performance"] = test_set_performance

    return ModelOutput(
        model_name=model_name,
        predictions=predictions,
        top_class=top_class,
        top_confidence=top_confidence,
        requires_mask=False,
        metadata=metadata,
    )


def _parse_positive_int(value: Any, default: Optional[int]) -> Optional[int]:
    """解析正整数配置；非法值回退到 default。"""
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _resolve_unified_batch_sizes(paths_config: dict) -> dict[str, Optional[int]]:
    """
    解析统一 batch_size 默认配置。
    优先级：模型内 batch_size > 顶层 batch_size.<model_type> > 内置默认。
    """
    raw = paths_config.get("batch_size", {}) if isinstance(paths_config, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "dino_unet": _parse_positive_int(raw.get("dino_unet"), 1),
        "autogluon": _parse_positive_int(raw.get("autogluon"), None),
        "llnm": _parse_positive_int(raw.get("llnm"), 4),
        "pathology_subtype_resnet": _parse_positive_int(
            raw.get("pathology_subtype_resnet"),
            32,
        ),
    }


def _resolve_global_device(paths_config: dict) -> str:
    """
    解析统一 device 配置。
    仅支持顶层 device（字符串），不兼容其他写法。
    """
    if not isinstance(paths_config, dict):
        raise ValueError("配置格式错误：根配置必须是字典")

    raw = paths_config.get("device", None)
    if raw is None:
        raise ValueError("配置缺失：请在 config 中设置顶层 device，例如 device: \"cuda\"")
    if not isinstance(raw, str):
        raise ValueError("配置格式错误：device 必须是字符串，例如 device: \"cuda:0\"")

    s = raw.strip()
    if not s:
        raise ValueError("配置格式错误：device 不能为空字符串")
    return s


def main(config_path: str = "config/config2.yaml"):
    print("=" * 70)
    print("甲状腺结节分类 Agent - 四类模型统一候选")
    print("=" * 70)
    print()

    # 1. 加载配置（包含所有配置信息）
    print(">>> 加载配置")
    config = load_config(config_path)
    if config is None:
        print("✗ 请先配置 config/config.yaml 文件")
        return

    # 使用统一的 config 作为 paths_config（所有配置都在 config.yaml 中）
    paths_config = config
    model_groups = resolve_pipeline_model_configs(paths_config)
    dino_models_config = model_groups["dino_unet"]
    autogluon_models_config = model_groups["autogluon"]
    llnm_models_config = model_groups["llnm"]
    pathology_resnet_models_config = model_groups["pathology_subtype_resnet"]
    unified_batch_sizes = _resolve_unified_batch_sizes(paths_config)
    global_device = _resolve_global_device(paths_config)
    unified_num_classes = 2
    print(f"✓ 全局推理设备: {global_device}")

    # 统一模型默认 batch_size/device/num_classes 注入。
    for cfg in llnm_models_config:
        if "batch_size" not in cfg and unified_batch_sizes.get("llnm") is not None:
            cfg["batch_size"] = int(unified_batch_sizes["llnm"])
        cfg["device"] = global_device
        cfg["num_classes"] = unified_num_classes

    for cfg in pathology_resnet_models_config:
        if "batch_size" not in cfg and unified_batch_sizes.get("pathology_subtype_resnet") is not None:
            cfg["batch_size"] = int(unified_batch_sizes["pathology_subtype_resnet"])
        cfg["device"] = global_device
        cfg["num_classes"] = unified_num_classes

    print("✓ 配置加载成功\n")

    # 2. 检查标签文件（优先使用 config.data.label_file，未配置则自动推断）
    print(">>> 检查标签文件")
    resolved_label_path, label_path_source = resolve_label_path(paths_config)
    if resolved_label_path is not None:
        if label_path_source == "config":
            print(f"✓ 已扫描到标签文件（来自配置）: {resolved_label_path}\n")
        else:
            print(f"✓ 已扫描到标签文件（自动推断）: {resolved_label_path}\n")
    else:
        configured_label = paths_config.get("data", {}).get("label_file", None)
        if configured_label is None or str(configured_label).strip().lower() == "null":
            print("⚠️ 未扫描到标签文件（config.data.label_file=null，且自动推断未命中）\n")
        else:
            print(f"⚠️ 未扫描到标签文件（配置路径不存在）: {configured_label}\n")

    # 3. 创建模型注册表
    registry = ModelRegistry()
    model_batch_sizes: dict[str, Optional[int]] = {}

    # 4. 注册 DINO-UNet 模型（支持多个）
    print(">>> 注册 DINO-UNet 多任务模型")

    if not dino_models_config:
        print("⚠️  没有配置 DINO-UNet 模型\n")
    else:
        print(f"    找到 {len(dino_models_config)} 个 DINO-UNet 模型配置")

        for idx, model_config in enumerate(dino_models_config, 1):
            model_name = model_config.get('name', f'dino_unet_{idx}')
            model_path = model_config['model_path']
            use_tirads = model_config.get('use_tirads', False)

            print(f"\n    [{idx}] {model_name}")
            print(f"        路径: {model_path}")
            print(f"        TI-RADS: {use_tirads}")

            if not os.path.exists(model_path):
                print(f"        ⚠️  模型路径不存在，跳过")
                continue

            try:
                dino_model = DINOUNetModel(
                    model_path=model_path,
                    device=global_device,
                    use_tirads=use_tirads
                )
                dino_model.load_model()

                # 如果有自定义名称，更新模型名称
                if 'name' in model_config:
                    dino_model.model_name = f"DINO_UNet_{model_name}"

                # DINO 默认逐张推理；可通过配置 batch_size 控制分块大小。
                dino_batch_size = _parse_positive_int(
                    model_config.get('batch_size', None),
                    unified_batch_sizes.get("dino_unet"),
                )

                # 保存数据集规模信息（需要先保存，因为推导training_data_devices需要用到）
                dataset_info = model_config.get('dataset_info', None)
                if dataset_info:
                    dino_model.dataset_info = dataset_info

                # 保存训练数据设备信息（仅保留显式配置）
                training_data_devices = model_config.get('training_data_devices', None)
                dino_model.training_data_devices = training_data_devices

                # 保存验证集性能指标
                validation_metrics = model_config.get('validation_metrics', None)
                if validation_metrics:
                    dino_model.validation_metrics = validation_metrics

                # 保存多个测试集上的性能（仅支持 test_set_performance）
                test_set_performance = model_config.get('test_set_performance', None)
                if test_set_performance:
                    dino_model.test_set_performance = test_set_performance

                registry.register_model(dino_model)
                model_batch_sizes[dino_model.model_name] = dino_batch_size
                print(f"        设备: {global_device}")
                print(f"        ✓ 注册成功")
            except Exception as e:
                print(f"        ✗ 注册失败: {e}")

        print()

    # 5. 注册 AutoGluon-PyRadiomics 模型（支持多个）
    print(">>> 注册 AutoGluon-PyRadiomics 模型")

    if not autogluon_models_config:
        print("⚠️  没有配置 AutoGluon 模型\n")
    else:
        print(f"    找到 {len(autogluon_models_config)} 个 AutoGluon 模型配置")

        for idx, model_config in enumerate(autogluon_models_config, 1):
            model_name = model_config.get('name', f'autogluon_{idx}')
            model_dir = model_config['model_dir']

            print(f"\n    [{idx}] {model_name}")
            print(f"        目录: {model_dir}")

            if not os.path.exists(model_dir):
                print(f"        ⚠️  模型目录不存在，跳过")
                continue

            try:
                autogluon_model = AutoGluonRadiomicsModel(
                    model_dir=model_dir
                )
                autogluon_model.load_model()

                # 如果有自定义名称，更新模型名称
                if 'name' in model_config:
                    autogluon_model.model_name = f"AutoGluon_{model_name}"

                # AutoGluon 默认一次处理全部样本；配置 batch_size 时按分块执行。
                autogluon_batch_size = _parse_positive_int(
                    model_config.get('batch_size', None),
                    unified_batch_sizes.get("autogluon"),
                )

                # 保存数据集规模信息（需要先保存，因为推导training_data_devices需要用到）
                dataset_info = model_config.get('dataset_info', None)
                if dataset_info:
                    autogluon_model.dataset_info = dataset_info

                # 保存训练数据设备信息（仅保留显式配置）
                training_data_devices = model_config.get('training_data_devices', None)
                autogluon_model.training_data_devices = training_data_devices

                # 保存验证集性能指标
                validation_metrics = model_config.get('validation_metrics', None)
                if validation_metrics:
                    autogluon_model.validation_metrics = validation_metrics

                # 保存多个测试集上的性能（仅支持 test_set_performance）
                test_set_performance = model_config.get('test_set_performance', None)
                if test_set_performance:
                    autogluon_model.test_set_performance = test_set_performance

                registry.register_model(autogluon_model)
                model_batch_sizes[autogluon_model.model_name] = autogluon_batch_size
                print(f"        ✓ 注册成功")
            except Exception as e:
                print(f"        ✗ 注册失败: {e}")

        print()

    if len(registry.list_models()) == 0:
        print("没有可用的模型，退出。")
        return

    project_root = Path(__file__).resolve().parent
    try:
        cal_map = load_calibration_map_from_config(config, project_root)
        registry.calibration_map = cal_map if cal_map else None
        if cal_map:
            print(
                f">>> 已加载概率校准表: {len(cal_map)} 个模型 {list(cal_map.keys())}\n"
            )
    except Exception:
        registry.calibration_map = None

    # 6. 初始化 Agent LLM（可配置开关，见 config agent_llm）
    print(">>> 初始化 Agent LLM")
    agent_config = config.get('agent', {})
    max_batch_size = agent_config.get('max_batch_size', 10)  # 从config读取，默认值为10
    binary_semantics = paths_config.get('binary_semantics', {}) if isinstance(paths_config, dict) else {}
    # 开关：是否启用 Agent 决策（默认为 True）
    enable_agent = agent_config.get('enable_agent', True)
    top_k = max(1, int(agent_config.get('top_k', 1)))

    agent = None
    if enable_agent:
        agent = LLMClassificationAgent(
            api_key=config['agent_llm']['api_key'],
            model_name=config['agent_llm']['model_name'],
            temperature=config['agent_llm']['temperature'],
            max_batch_size=max_batch_size,
            selection_mode=agent_config.get('selection_mode', 'deterministic'),
            top_k=top_k,
        )
        if isinstance(binary_semantics, dict) and binary_semantics:
            class_0 = str(binary_semantics.get('class_0', 'class_0'))
            class_1 = str(binary_semantics.get('class_1', 'class_1'))
            pos_idx = int(binary_semantics.get('positive_class_index', 1))
            agent.system_prompt += (
                "\n\n【当前二分类任务语义】"
                f"\n- class_0: {class_0}"
                f"\n- class_1: {class_1}"
                f"\n- positive_class_index: {pos_idx}"
                "\n请严格按以上语义理解 0/1 输出并进行决策。"
            )
        print(f"✓ Agent LLM 初始化成功 (max_batch_size={max_batch_size}, top_k={top_k})\n")
    else:
        print(
            f"⚠️ 已在配置中关闭 Agent LLM，将按 top_confidence 取前 {top_k} 个模型做 soft voting。\n"
        )

    # 7. 创建图像处理器
    image_processor = ImageProcessor()

    # 8. 获取图像路径（文件或目录）
    print(">>> 加载图像路径")
    image_input = paths_config['data']['image_input']

    print(f"    图像路径: {image_input}")

    if not os.path.exists(image_input):
        print(f"✗ 路径不存在: {image_input}")
        print("   请检查 config/config.yaml 中的 data.image_input 配置")
        return

    # 判断是文件还是目录
    image_input_path = Path(image_input)
    if image_input_path.is_file():
        image_files = [image_input_path]
        print(f"✓ 检测到单个图像文件")
    elif image_input_path.is_dir():
        image_files = get_image_files(image_input)
        print(f"✓ 检测到图像目录，找到 {len(image_files)} 个图像文件")
    else:
        print(f"✗ 无效的路径类型")
        return

    if len(image_files) == 0:
        print("✗ 未找到任何图像文件")
        return

    # 9. 获取掩码路径（如果需要）
    mask_input = None
    mask_dir = None

    # 检查是否有任何模型需要掩码
    needs_mask = any(model.requires_mask for model in [registry.get_model(name) for name in registry.list_models()])

    if needs_mask:
        mask_input = paths_config['data'].get('mask_input', '')

        print(f">>> 加载掩码路径")
        print(f"    掩码路径: {mask_input if mask_input else '(未配置)'}")

        if mask_input and os.path.exists(mask_input):
            mask_input_path = Path(mask_input)
            if mask_input_path.is_dir():
                mask_dir = mask_input_path
                print(f"✓ 检测到掩码目录")
            else:
                print(f"✓ 检测到单个掩码文件")
        elif mask_input:
            print(f"⚠️  掩码路径不存在: {mask_input}")
            print("   需要掩码的模型将无法运行")
        else:
            print(f"⚠️  未配置掩码路径，需要掩码的模型将无法运行")

    print()

    # 10. 批量推理优化流程
    print("=" * 70)
    print(f"开始批量处理 ({len(image_files)} 个图像)")
    print("=" * 70)
    print()

    # 步骤1: 加载所有图像和掩码
    print(">>> 步骤 1/3: 加载所有图像和掩码")
    image_data = []  # 存储 (image_file, image, mask)

    for idx, image_file in enumerate(image_files, 1):
        print(f"  [{idx}/{len(image_files)}] 加载 {image_file.name}...", end=" ")

        try:
            # 加载图像
            image = image_processor.load_image(str(image_file))

            # 加载对应的掩码
            mask = None
            if mask_dir:
                mask_file = find_corresponding_mask(image_file, mask_dir)
                if mask_file:
                    mask = image_processor.load_mask(str(mask_file))
            elif mask_input and Path(mask_input).is_file() and len(image_files) == 1:
                mask = image_processor.load_mask(mask_input)

            image_data.append((image_file, image, mask))
            print("✓")

        except Exception as e:
            print(f"✗ 失败: {e}")

    print(f"✓ 成功加载 {len(image_data)} 个图像\n")

    if len(image_data) == 0:
        print("✗ 没有成功加载的图像，退出")
        return

    # 步骤2: 对每个模型，批量推理所有图像
    print(">>> 步骤 2/3: 使用每个模型对所有图像进行推理")

    # 数据结构: all_predictions[image_index][model_name] = PredictionResult
    all_predictions = [{} for _ in range(len(image_data))]

    for model_name in registry.list_models():
        model = registry.get_model(model_name)

        print(f"\n  模型: {model.model_name}")
        print(f"  {'='*65}")

        # 准备批量数据（过滤掉缺少掩码的图像）
        batch_images = []
        batch_masks = []
        batch_indices = []

        for idx, (image_file, image, mask) in enumerate(image_data):
            # 检查是否需要掩码
            if model.requires_mask and mask is None:
                print(f"    [{idx+1}/{len(image_data)}] {image_file.name}... 跳过 (缺少掩码)")
                continue

            batch_images.append(image)
            batch_masks.append(mask)
            batch_indices.append(idx)

        if len(batch_images) == 0:
            print(f"  ⚠️  没有可用的图像（可能缺少掩码），跳过此模型\n")
            continue

        configured_batch_size = model_batch_sizes.get(model.model_name)
        if configured_batch_size is None:
            configured_batch_size = len(batch_images)
        configured_batch_size = max(1, int(configured_batch_size))
        if configured_batch_size < len(batch_images):
            print(f"  使用 batch_size={configured_batch_size} 分块推理")

        # 使用批量预测（如果模型支持）
        try:
            # 检查模型是否实现了优化的批量预测
            # 检查是否是 AutoGluon 模型（通过模型名称或类型判断）
            is_autogluon = (
                hasattr(model, 'predict_batch') and
                (model.model_name.startswith("AutoGluon_") or
                 model.model_name == "AutoGluon_PyRadiomics" or
                 isinstance(model, AutoGluonRadiomicsModel))
            )

            results: list[Optional[ModelOutput]] = [None] * len(batch_images)
            for start in range(0, len(batch_images), configured_batch_size):
                end = min(start + configured_batch_size, len(batch_images))
                chunk_images = batch_images[start:end]
                chunk_masks = batch_masks[start:end]

                if is_autogluon:
                    # 使用优化的批量预测。
                    # 注意：PyRadiomics 特征提取内部仍逐张，但 AutoGluon 概率预测可批量执行。
                    chunk_results = model.predict_batch(
                        chunk_images,
                        chunk_masks,
                        show_progress=True,
                    )
                    for j, r in enumerate(chunk_results):
                        if r is not None:
                            try:
                                maybe_apply_calibration_map(r, registry.calibration_map)
                            except Exception:
                                pass
                        results[start + j] = r
                else:
                    # DINO 等模型按分块逐张处理。
                    for j, (image, mask) in enumerate(zip(chunk_images, chunk_masks)):
                        global_idx = start + j
                        image_file = image_data[batch_indices[global_idx]][0]
                        print(f"    [{global_idx+1}/{len(batch_images)}] {image_file.name}...", end=" ")

                        try:
                            result = model.predict(image, mask)
                            try:
                                maybe_apply_calibration_map(
                                    result, registry.calibration_map
                                )
                            except Exception:
                                pass
                            results[global_idx] = result
                            print(f"✓ {result.top_class} ({result.top_confidence:.4f})")
                        except Exception as e:
                            print(f"✗ 失败: {e}")
                            results[global_idx] = None

            # 保存结果
            for result, orig_idx in zip(results, batch_indices):
                if result is not None:
                    all_predictions[orig_idx][model_name] = result

            print(f"  ✓ {model.model_name} 完成所有推理\n")

        except Exception as e:
            print(f"  ✗ {model.model_name} 批量推理失败: {e}")
            import traceback
            traceback.print_exc()
            print()

    # 统一阶段：LLNM 二分类模型
    image_paths = [image_file for image_file, _, _ in image_data]
    if llnm_models_config:
        print("\n>>> 统一模型推理：LLNM 二分类")
        for idx, model_cfg in enumerate(llnm_models_config, 1):
            model_name = f"LLNM_{model_cfg.get('name', f'llnm_{idx}') }"
            test_set_performance = model_cfg.get("test_set_performance", None)
            print(f"  [{idx}/{len(llnm_models_config)}] {model_name}")
            try:
                pred_map = run_llnm_binary_for_image_paths(
                    image_paths=image_paths,
                    llnm_cfg=model_cfg,
                    project_root=project_root,
                )
                added = 0
                for data_idx, (image_file, _, _) in enumerate(image_data):
                    rec = pred_map.get(str(image_file.resolve()))
                    if not isinstance(rec, dict):
                        continue
                    out = _build_binary_model_output(
                        binary_record=rec,
                        model_name=model_name,
                        task_name="llnm_lymph_node_metastasis",
                        test_set_performance=test_set_performance,
                    )
                    try:
                        maybe_apply_calibration_map(out, registry.calibration_map)
                    except Exception:
                        pass
                    all_predictions[data_idx][model_name] = out
                    added += 1
                print(f"    ✓ 完成，获得 {added} 条预测")
            except Exception as e:
                print(f"    ✗ 失败: {e}")

    # 统一阶段：病理亚型 ResNet 二分类模型
    if pathology_resnet_models_config:
        print("\n>>> 统一模型推理：病理亚型 ResNet 二分类")
        for idx, model_cfg in enumerate(pathology_resnet_models_config, 1):
            model_name = f"ResNet_{model_cfg.get('name', f'pathology_subtype_resnet_{idx}') }"
            test_set_performance = model_cfg.get("test_set_performance", None)
            print(f"  [{idx}/{len(pathology_resnet_models_config)}] {model_name}")
            try:
                pred_map = run_resnet_binary_for_image_paths(
                    image_paths=image_paths,
                    resnet_cfg=model_cfg,
                    project_root=project_root,
                )
                added = 0
                for data_idx, (image_file, _, _) in enumerate(image_data):
                    rec = pred_map.get(str(image_file.resolve()))
                    if not isinstance(rec, dict):
                        continue
                    out = _build_binary_model_output(
                        binary_record=rec,
                        model_name=model_name,
                        task_name="pathology_subtype_resnet",
                        test_set_performance=test_set_performance,
                    )
                    try:
                        maybe_apply_calibration_map(out, registry.calibration_map)
                    except Exception:
                        pass
                    all_predictions[data_idx][model_name] = out
                    added += 1
                print(f"    ✓ 完成，获得 {added} 条预测")
            except Exception as e:
                print(f"    ✗ 失败: {e}")

    valid_count = sum(1 for preds in all_predictions if len(preds) > 0)
    print(f"\n✓ 当前共有 {valid_count} 个图像获得了可用预测\n")

    # 过滤掉没有任何预测结果的图像（主流程 + 辅助流程）
    valid_data = []
    for idx, (image_file, image, mask) in enumerate(image_data):
        if len(all_predictions[idx]) > 0:
            valid_data.append((idx, image_file, all_predictions[idx]))

    if len(valid_data) == 0:
        print("✗ 没有图像有成功的推理结果，退出")
        return

    print(f"✓ 共 {len(valid_data)} 个图像有成功的推理结果\n")

    # 步骤3: 决策与结果汇总（根据配置选择是否使用 Agent）
    print(">>> 步骤 3/3: 决策与结果汇总")
    print("=" * 70)

    # 提前生成结果文件路径，便于 Agent 批量时每批完成后增量写入同一文件
    output_dir = Path(paths_config['output'].get('output_dir', 'output'))
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    all_results = []

    if enable_agent and agent is not None:
        # 使用 Agent 进行综合决策
        # 构建批量预测数据
        batch_predictions = []
        for idx, image_file, predictions_dict in valid_data:
            batch_predictions.append({
                "image_file": str(image_file),
                "image_name": image_file.name,
                "predictions": list(predictions_dict.values())
            })

        # Agent 批量决策
        try:
            # 获取输入数据的设备信息
            input_device_info = paths_config['data'].get('device_info', None)
            if input_device_info:
                print(f"\n输入数据设备信息: {', '.join(input_device_info)}")
            input_data_info = paths_config.get('data', {})

            print(f"\n正在处理 {len(batch_predictions)} 个图像的综合决策（使用 Agent）...\n")
            batch_decisions = agent.select_best_model_batch(
                batch_predictions,
                input_device_info=input_device_info,
                input_data_info=input_data_info,
                incremental_save_path=str(output_file),
            )

            print("✓ Agent 批量决策完成!\n")

            # 显示结果
            print("=" * 70)
            print("决策结果（Agent 模式）")
            print("=" * 70)

            for i, (idx, image_file, predictions_dict) in enumerate(valid_data):
                decision = batch_decisions[i]

                print(f"\n[{i+1}/{len(valid_data)}] {image_file.name}")
                print(f"  选择模型: {decision.selected_model}")
                print(f"  预测类别: {decision.selected_class}")
                print(f"  置信度: {decision.confidence:.4f}")
                print(f"  理由: {decision.reasoning}")

                # 保存结果
                result_dict = {
                    "image_file": str(image_file),
                    "image_name": image_file.name,
                    "selected_model": decision.selected_model,
                    "predicted_class": decision.selected_class,
                    "confidence": float(decision.confidence),
                    "reasoning": decision.reasoning,
                    "all_predictions": [
                        {
                            "model": p.model_name,
                            "top_class": p.top_class,
                            "top_confidence": float(p.top_confidence),
                            "predictions": {k: float(v) for k, v in p.predictions.items()}
                        }
                        for p in predictions_dict.values()
                    ]
                }
                all_results.append(result_dict)

        except Exception as e:
            print(f"\n✗ Agent 批量决策失败: {e}")
            import traceback
            traceback.print_exc()

            # 如果批量决策失败，回退到单张决策
            print("\n尝试回退到单张决策模式（仍使用 Agent）...")

            for idx, image_file, predictions_dict in valid_data:
                try:
                    predictions = list(predictions_dict.values())
                    decision = agent.select_best_model(
                        predictions,
                        input_device_info=paths_config.get('data', {}).get('device_info', None),
                        input_data_info=paths_config.get('data', {})
                    )

                    result_dict = {
                        "image_file": str(image_file),
                        "image_name": image_file.name,
                        "selected_model": decision.selected_model,
                        "predicted_class": decision.selected_class,
                        "confidence": float(decision.confidence),
                        "reasoning": decision.reasoning,
                        "all_predictions": [
                            {
                                "model": p.model_name,
                                "top_class": p.top_class,
                                "top_confidence": float(p.top_confidence),
                                "predictions": {k: float(v) for k, v in p.predictions.items()}
                            }
                            for p in predictions
                        ]
                    }
                    all_results.append(result_dict)

                except Exception as e2:
                    print(f"✗ 单张决策也失败: {image_file.name}: {e2}")
    else:
        # 不使用 Agent：按 top_confidence 取前 top_k 个模型，再对类别概率取均值
        print("Agent LLM 已关闭，使用 top-k soft voting 策略进行综合决策。\n")

        print("=" * 70)
        print("决策结果（Soft Voting 模式）")
        print("=" * 70)

        for i, (idx, image_file, predictions_dict) in enumerate(valid_data):
            predictions = list(predictions_dict.values())
            sorted_preds = sorted(
                predictions, key=lambda p: p.top_confidence, reverse=True
            )
            k = min(top_k, len(sorted_preds))
            subset = sorted_preds[:k]

            avg_class_probs = _average_class_probabilities(subset)
            selected_class, best_confidence = _winning_class_from_avg_probs(avg_class_probs)

            selected_model_name = "soft_voting_topk_ensemble"

            reasoning = (
                f"未使用 Agent，按 top_confidence 取前 {k} 个模型后对类别概率取均值，"
                f"类别 '{selected_class}' 的平均概率最高（{best_confidence:.4f}）。"
            )

            print(f"\n[{i+1}/{len(valid_data)}] {image_file.name}")
            print(f"  选择模型: {selected_model_name}")
            print(f"  预测类别: {selected_class}")
            print(f"  置信度: {best_confidence:.4f}")
            print(f"  理由: {reasoning}")

            result_dict = {
                "image_file": str(image_file),
                "image_name": image_file.name,
                "selected_model": selected_model_name,
                "predicted_class": selected_class,
                "confidence": float(best_confidence),
                "reasoning": reasoning,
                "all_predictions": [
                    {
                        "model": p.model_name,
                        "top_class": p.top_class,
                        "top_confidence": float(p.top_confidence),
                        "predictions": {k: float(v) for k, v in p.predictions.items()}
                    }
                    for p in predictions
                ]
            }
            all_results.append(result_dict)

    # 11. 保存所有结果（Agent 批量模式下已按批增量写入 output_file，此处再次写入以统一格式并覆盖）
    if len(all_results) > 0:
        print("\n" + "=" * 70)
        print("保存结果")
        print("=" * 70)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"✓ 结果已保存到: {output_file}")
        print(f"  共处理 {len(all_results)} 个图像")

        # 统计结果
        print("\n【统计信息】")
        model_counts = {}
        class_counts = {}

        for result in all_results:
            model = result["selected_model"]
            pred_class = result["predicted_class"]

            model_counts[model] = model_counts.get(model, 0) + 1
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        print(f"\n模型选择统计:")
        for model, count in model_counts.items():
            print(f"  {model}: {count} 次 ({count/len(all_results)*100:.1f}%)")

        print(f"\n类别预测统计:")
        for pred_class, count in class_counts.items():
            print(f"  {pred_class}: {count} 次 ({count/len(all_results)*100:.1f}%)")

        # 12. 计算平均分类指标（需要标签文件）
        # 仅在 labels 可推断且存在时进行评估，否则给出提示。
        decision_threshold = float(paths_config.get("agent", {}).get("decision_threshold", 0.5))

        def _load_labels(label_path: Path) -> dict[str, int]:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            labels: dict[str, int] = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename") or item.get("image_name")
                    mal = item.get("malignancy", item.get("label"))
                    if fn is None or mal is None:
                        continue
                    labels[str(fn)] = int(mal)
            elif isinstance(data, dict):
                for k, v in data.items():
                    labels[str(k)] = int(v)
            return labels

        def _label_lookup(labels: dict[str, int], image_name: str):
            # 直接匹配
            if image_name in labels:
                return labels[image_name]

            # 常见情况：结果里叫 "TN3K_test_0000.jpg"；labels 里可能是 "0000.jpg"
            ext = Path(image_name).suffix
            stem = Path(image_name).stem
            tokens = stem.split("_")
            if len(tokens) >= 2:
                candidate = tokens[-1] + ext
                if candidate in labels:
                    return labels[candidate]
            return None

        def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
            # ECE: sum_{bins} (bin_prob * |acc_bin - conf_bin|)
            y_true = y_true.astype(np.float64)
            y_prob = y_prob.astype(np.float64)
            n = y_true.shape[0]
            if n == 0:
                return float("nan")
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            # digitize: bins are [edge[i], edge[i+1]) except the last edge inclusive
            bin_ids = np.digitize(y_prob, bin_edges, right=False) - 1
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

            ece = 0.0
            for b in range(n_bins):
                mask = bin_ids == b
                if not np.any(mask):
                    continue
                conf_bin = float(np.mean(y_prob[mask]))
                acc_bin = float(np.mean(y_true[mask]))  # y_true is 0/1
                bin_prob = float(np.mean(mask))
                ece += bin_prob * abs(acc_bin - conf_bin)
            return float(ece)

        def _compute_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
            y_pred_bin = (y_prob >= threshold).astype(np.int32)
            acc = float(np.mean((y_pred_bin == y_true).astype(np.float64)))
            prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
            rec = float(recall_score(y_true, y_pred_bin, zero_division=0))
            f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

            tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
            fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            auroc = float("nan")
            auprc = float("nan")
            if len(np.unique(y_true)) > 1:
                try:
                    auroc = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    pass
                try:
                    auprc = float(average_precision_score(y_true, y_prob))
                except Exception:
                    pass

            ece = _ece_binary(y_true, y_prob, n_bins=10)
            return {
                "auroc": auroc,
                "auprc": auprc,
                "acc": acc,
                "prec": prec,
                "recall": rec,
                "f1": f1,
                "specificity": specificity,
                "ece": ece,
            }

        def _compute_bootstrap_ci95(
            y_true: np.ndarray,
            y_prob: np.ndarray,
            threshold: float,
            n_boot: int = 2000,
            seed: int = 0,
        ) -> dict[str, dict[str, Optional[float]]]:
            rng = np.random.default_rng(seed)
            n = len(y_true)
            metric_keys = ["auroc", "auprc", "acc", "prec", "recall", "f1", "specificity", "ece"]
            samples = {k: [] for k in metric_keys}

            for _ in range(max(1, n_boot)):
                idx = rng.integers(0, n, n)
                yt = y_true[idx]
                yp = y_prob[idx]
                m = _compute_point_metrics(yt, yp, threshold)
                for k in metric_keys:
                    samples[k].append(float(m[k]) if not np.isnan(m[k]) else np.nan)

            out: dict[str, dict[str, Optional[float]]] = {}
            for k in metric_keys:
                arr = np.asarray(samples[k], dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    out[k] = {"mean": None, "ci95_lower": None, "ci95_upper": None}
                    continue
                out[k] = {
                    "mean": float(np.mean(valid)),
                    "ci95_lower": float(np.percentile(valid, 2.5)),
                    "ci95_upper": float(np.percentile(valid, 97.5)),
                }
            return out

        try:
            label_path = resolved_label_path
            if label_path is None or not label_path.exists():
                print("\n【平均分类指标】跳过：未找到可用 labels 文件")
            else:
                labels = _load_labels(label_path)

                # 对齐 results 与 labels，构造 y_true / y_prob
                y_true_list: list[int] = []
                y_prob_list: list[float] = []
                for r in all_results:
                    image_name = r.get("image_name") or ""
                    gt = _label_lookup(labels, image_name)
                    if gt is None:
                        continue

                    pred_class = str(r.get("predicted_class", "")).strip()
                    conf = float(r.get("confidence", 0.5))
                    # 统一成 P(class_1)
                    pred_lower = pred_class.lower()
                    if pred_class == "1" or pred_lower == "1":
                        p_class_1 = conf
                    elif pred_class == "0" or pred_lower == "0":
                        p_class_1 = 1.0 - conf
                    else:
                        p_class_1 = 0.5

                    y_true_list.append(gt)
                    y_prob_list.append(p_class_1)

                y_true = np.asarray(y_true_list, dtype=np.int32)
                y_prob = np.asarray(y_prob_list, dtype=np.float64)
                n_eval = int(y_true.shape[0])

                if n_eval == 0:
                    print("\n【平均分类指标】跳过：results 与 labels 无对齐样本")
                else:
                    n_boot = int(paths_config.get("agent", {}).get("metrics_n_boot", 2000))
                    boot_seed = int(paths_config.get("agent", {}).get("metrics_bootstrap_seed", 0))
                    point = _compute_point_metrics(y_true, y_prob, decision_threshold)
                    ci95 = _compute_bootstrap_ci95(
                        y_true=y_true,
                        y_prob=y_prob,
                        threshold=decision_threshold,
                        n_boot=n_boot,
                        seed=boot_seed,
                    )

                    print("\n【平均分类指标】")
                    print(f"  样本数: {n_eval}")
                    print(f"  Bootstrap: n_boot={n_boot}, seed={boot_seed}")
                    pretty_names = [
                        ("AUROC", "auroc"),
                        ("AUPRC", "auprc"),
                        ("Acc", "acc"),
                        ("Prec", "prec"),
                        ("Recall", "recall"),
                        ("F1", "f1"),
                        ("Specificity", "specificity"),
                        ("ECE", "ece"),
                    ]
                    for display, key in pretty_names:
                        v = point[key]
                        c = ci95[key]
                        if v is None or np.isnan(v) or c["mean"] is None:
                            print(f"  {display}: N/A")
                        else:
                            print(
                                f"  {display}: {float(v):.4f}  "
                                f"(mean={float(c['mean']):.4f}, CI95=[{float(c['ci95_lower']):.4f}, {float(c['ci95_upper']):.4f}])"
                            )

                    metrics_out = {
                        "label_path": str(label_path),
                        "n_samples": n_eval,
                        "threshold": decision_threshold,
                        "n_boot": n_boot,
                        "metrics": {k: (None if np.isnan(v) else round(float(v), 6)) for k, v in point.items()},
                        "metrics_ci95": {
                            k: {
                                "mean": None if v["mean"] is None else round(float(v["mean"]), 6),
                                "ci95_lower": None if v["ci95_lower"] is None else round(float(v["ci95_lower"]), 6),
                                "ci95_upper": None if v["ci95_upper"] is None else round(float(v["ci95_upper"]), 6),
                            }
                            for k, v in ci95.items()
                        },
                    }
                    metrics_out_path = output_dir / f"agent_metrics_{timestamp}.json"
                    with open(metrics_out_path, "w", encoding="utf-8") as f:
                        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
                    print(f"  指标已保存: {metrics_out_path}")
        except Exception as e:
            print(f"\n【平均分类指标】计算失败（已跳过）：{e}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thyroid classification pipeline with configurable YAML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config2.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
