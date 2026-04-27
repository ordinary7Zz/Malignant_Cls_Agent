"""
Main2 专用分类 Agent。

设计目标：
- 不依赖 base_datasets_info 或任何基础数据集元信息。
- 支持 4 类模型输出（DINO / AutoGluon / LLNM / ResNet）统一融合。
- 保持与现有 main2 调用接口兼容。
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models.base_model import ModelOutput


def _average_class_probabilities(predictions: List[ModelOutput]) -> Dict[str, float]:
    """对多个模型的各类别概率取算术平均（缺失类别视为 0）。"""
    if not predictions:
        return {}
    class_sums: Dict[str, float] = {}
    for prediction in predictions:
        for class_name, prob in prediction.predictions.items():
            class_sums[class_name] = class_sums.get(class_name, 0.0) + float(prob)
    count = len(predictions)
    return {class_name: float(v) / count for class_name, v in class_sums.items()}


def _winning_class_from_avg_probs(avg_probs: Dict[str, float]) -> tuple[str, float]:
    """从平均概率中选出最高类别。"""
    if not avg_probs:
        return "", 0.0
    return max(avg_probs.items(), key=lambda x: x[1])


@dataclass
class AgentDecision:
    """Agent 对最终输出的决策结果。"""

    selected_model: str
    selected_class: str
    confidence: float
    reasoning: str
    all_predictions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "selected_class": self.selected_class,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "all_predictions": self.all_predictions,
        }


class LLMClassificationAgent:
    """
    Main2 专用多模型融合 Agent。

    决策信息源仅包括：
    - 模型输出概率（含 top_confidence）
    - 不确定性（entropy / margin_top2）
    - 验证指标（accuracy / auc / f1_score）
    - 模型间一致性（num_models_same_class / vote_entropy）
    - 输入上下文（data.device_info / image_input / mask_input / label_file）
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_batch_size: int = 10,
        selection_mode: str = "deterministic",
        top_k: int = 1,
    ):
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError("GLM_API_KEY is not set and no api_key was provided.")

        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_batch_size = max(1, int(max_batch_size))
        self.selection_mode = selection_mode
        self.top_k = max(1, int(top_k))

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.system_prompt = """你是甲状腺超声二分类多模型融合专家，需要在候选模型中选出最可信的一项（或 top_k 组合）。

【禁止信息源】不可使用任何基础数据集映射/来源推断（如 TN3K/ThyroidXL/TN5K/CineClip 的设备映射）。

【可用字段】
1) 主置信度：优先 metadata.classification_uncertainty.top_confidence_calibrated，否则 top_confidence_raw 或 top_confidence。
2) 不确定性：entropy 越低越好，margin_top2 越高越好。
3) 验证指标：validation_metrics.on_training_dataset 的 accuracy / auc / f1_score。
4) 一致性：num_models_same_class 越高越好，vote_entropy 越低越好。

【决策序】
R1 主置信度优先；
R2 若主置信度差 < 0.05，比不确定性（entropy、margin_top2）；
R3 若仍接近，比验证指标（AUC > ACC > F1）；
R4 若仍接近，比一致性（num_models_same_class、vote_entropy）；
R5 若仍接近，选择更稳健的概率分布（避免极端波动）。

【输出】只输出纯 JSON（无 Markdown/代码块/解释前缀）。
top_k=1 时字段：selected_model, selected_class, confidence, reasoning。
top_k>1 时字段：selected_models(长度=top_k), reasoning。

reasoning 必须 2~5 句中文，包含关键数值（至少：主置信度 + entropy 或 margin_top2 的比较）。"""

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    text = part.strip()
                    break

        start_idx = text.find("{")
        if start_idx < 0:
            return text.strip()

        brace_count = 0
        in_string = False
        escape_next = False
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx : idx + 1]
        return text[start_idx:].strip()

    @staticmethod
    def _predictions_top_class_unanimous(predictions: List[ModelOutput]) -> bool:
        if not predictions:
            return False
        first_class = predictions[0].top_class
        return all(pred.top_class == first_class for pred in predictions)

    @staticmethod
    def _read_primary_confidence(prediction_dict: Dict[str, Any]) -> float:
        metadata = prediction_dict.get("metadata") or {}
        cu = metadata.get("classification_uncertainty") or {}
        value = cu.get("top_confidence_calibrated")
        if value is None:
            value = cu.get("top_confidence_raw")
        if value is None:
            value = prediction_dict.get("top_confidence", 0.0)
        return float(value)

    @staticmethod
    def _read_validation_metrics(prediction_dict: Dict[str, Any]) -> tuple[float, float, float]:
        metadata = prediction_dict.get("metadata") or {}
        on_train = (metadata.get("validation_metrics") or {}).get("on_training_dataset") or {}
        acc = float(on_train.get("accuracy") or 0.0)
        auc = float(on_train.get("auc") or 0.0)
        f1 = float(on_train.get("f1_score") or 0.0)
        return acc, auc, f1

    @staticmethod
    def _read_uncertainty(prediction_dict: Dict[str, Any]) -> tuple[float, float]:
        metadata = prediction_dict.get("metadata") or {}
        cu = metadata.get("classification_uncertainty") or {}
        entropy = cu.get("entropy")
        margin = cu.get("margin_top2")

        if entropy is None or margin is None:
            probs = list((prediction_dict.get("predictions") or {}).values())
            if probs:
                total = float(sum(probs))
                if total > 0:
                    norm_probs = [float(p) / total for p in probs]
                    entropy = -sum(p * math.log(p + 1e-12, 2) for p in norm_probs if p > 0)
                sorted_probs = sorted((float(p) for p in probs), reverse=True)
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]

        return float(entropy or 0.0), float(margin or 0.0)

    def _build_compact_prediction_dicts(self, predictions: List[ModelOutput]) -> List[Dict[str, Any]]:
        votes_per_class: Dict[str, int] = {}
        for prediction in predictions:
            votes_per_class[prediction.top_class] = votes_per_class.get(prediction.top_class, 0) + 1

        total_models = len(predictions)
        vote_entropy = 0.0
        if total_models > 0:
            probs = [count / total_models for count in votes_per_class.values()]
            vote_entropy = -sum(p * math.log(p + 1e-12, 2) for p in probs if p > 0)

        compact_predictions: List[Dict[str, Any]] = []
        for prediction in predictions:
            pred_dict = prediction.to_dict()
            metadata = pred_dict.get("metadata") or {}
            full_probs = pred_dict.get("predictions") or {}

            entropy, margin_top2 = self._read_uncertainty(pred_dict)
            on_train = (metadata.get("validation_metrics") or {}).get("on_training_dataset") or {}

            compact_predictions.append(
                {
                    "model_name": pred_dict.get("model_name"),
                    "top_class": pred_dict.get("top_class"),
                    "top_confidence": float(pred_dict.get("top_confidence") or 0.0),
                    "predictions": full_probs,
                    "metadata": {
                        "classification_uncertainty": {
                            **(
                                {
                                    "top_confidence_calibrated": (metadata.get("classification_uncertainty") or {}).get(
                                        "top_confidence_calibrated"
                                    )
                                }
                                if (metadata.get("classification_uncertainty") or {}).get("top_confidence_calibrated")
                                is not None
                                else {}
                            ),
                            "top_confidence_raw": float(pred_dict.get("top_confidence") or 0.0),
                            "entropy": float(entropy),
                            "margin_top2": float(margin_top2),
                        },
                        "validation_metrics": {
                            "on_training_dataset": {
                                "accuracy": on_train.get("accuracy"),
                                "auc": on_train.get("auc"),
                                "f1_score": on_train.get("f1_score"),
                            }
                        },
                        "consistency_metrics": {
                            "num_models_same_class": votes_per_class.get(pred_dict.get("top_class"), 0),
                            "total_models": total_models,
                            "vote_entropy": vote_entropy,
                        },
                    },
                }
            )

        return compact_predictions

    @staticmethod
    def _format_input_data_info_text(
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = []
        data_info = input_data_info or {}

        def is_unknown(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return value.strip() == "" or value.strip().lower() == "null"
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) == 0
            return False

        device_info = input_device_info
        if is_unknown(device_info):
            device_info = data_info.get("device_info")

        if is_unknown(device_info):
            lines.append("- device_info: 未知")
        elif isinstance(device_info, list):
            lines.append(f"- device_info: {', '.join(str(item) for item in device_info)}")
        else:
            lines.append(f"- device_info: {device_info}")

        for key in ["image_input", "mask_input", "label_file"]:
            value = data_info.get(key)
            if is_unknown(value):
                lines.append(f"- {key}: 未知")
            else:
                lines.append(f"- {key}: {value}")

        return "\n输入数据上下文（data；null=未知）:\n" + "\n".join(lines) + "\n"

    def _fallback_single(self, predictions: List[ModelOutput]) -> AgentDecision:
        pred_dicts = self._build_compact_prediction_dicts(predictions)
        scored: List[tuple[float, Dict[str, Any]]] = []

        for item in pred_dicts:
            primary_conf = self._read_primary_confidence(item)
            entropy, margin = self._read_uncertainty(item)
            acc, auc, f1 = self._read_validation_metrics(item)
            consistency = (item.get("metadata") or {}).get("consistency_metrics") or {}
            num_same = float(consistency.get("num_models_same_class") or 0.0)
            total = float(consistency.get("total_models") or 1.0)
            vote_entropy = float(consistency.get("vote_entropy") or 0.0)
            consensus_ratio = num_same / total if total > 0 else 0.0

            score = (
                1.00 * primary_conf
                + 0.15 * margin
                - 0.10 * entropy
                + 0.08 * auc
                + 0.05 * acc
                + 0.05 * f1
                + 0.03 * consensus_ratio
                - 0.02 * vote_entropy
            )
            scored.append((float(score), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_item = scored[0][1]
        selected_model = str(best_item.get("model_name"))
        selected_class = str(best_item.get("top_class"))
        confidence = self._read_primary_confidence(best_item)
        entropy, margin = self._read_uncertainty(best_item)

        reasoning = (
            "降级策略：基于主置信度、不确定性、验证指标与一致性综合打分。"
            f" 选中模型 {selected_model}，主置信度={confidence:.4f}，entropy={entropy:.4f}，margin_top2={margin:.4f}。"
        )
        return AgentDecision(
            selected_model=selected_model,
            selected_class=selected_class,
            confidence=float(confidence),
            reasoning=reasoning,
            all_predictions=[prediction.to_dict() for prediction in predictions],
        )

    def _fallback_topk(self, predictions: List[ModelOutput]) -> AgentDecision:
        scored = []
        for prediction in predictions:
            pred_dict = prediction.to_dict()
            primary_conf = self._read_primary_confidence(pred_dict)
            entropy, margin = self._read_uncertainty(pred_dict)
            score = 1.0 * primary_conf + 0.15 * margin - 0.10 * entropy
            scored.append((score, prediction))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(self.top_k, len(scored))
        subset = [item[1] for item in scored[:k]]
        avg_probs = _average_class_probabilities(subset)
        selected_class, confidence = _winning_class_from_avg_probs(avg_probs)

        reasoning = (
            f"降级策略：按无数据集先验综合分数选取前 {k} 个模型并做概率平均，"
            f"得到类别「{selected_class}」平均概率最高（{confidence:.4f}）。"
        )
        return AgentDecision(
            selected_model="agent_topk_soft_voting",
            selected_class=selected_class,
            confidence=float(confidence),
            reasoning=reasoning,
            all_predictions=[prediction.to_dict() for prediction in predictions],
        )

    def select_best_model(
        self,
        predictions: List[ModelOutput],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        if not predictions:
            raise ValueError("No predictions provided")

        if self._predictions_top_class_unanimous(predictions):
            best = max(predictions, key=lambda prediction: prediction.top_confidence)
            return AgentDecision(
                selected_model=best.model_name,
                selected_class=best.top_class,
                confidence=float(best.top_confidence),
                reasoning=(
                    f"所有模型 top_class 一致为「{best.top_class}」，"
                    f"直接选取置信度最高模型 {best.model_name}（{best.top_confidence:.4f}）。"
                ),
                all_predictions=[prediction.to_dict() for prediction in predictions],
            )

        if self.top_k > 1:
            return self._fallback_topk(predictions)

        compact_predictions = self._build_compact_prediction_dicts(predictions)
        payload = {
            "num_models": len(compact_predictions),
            "predictions": compact_predictions,
        }

        data_info_text = self._format_input_data_info_text(
            input_device_info=input_device_info,
            input_data_info=input_data_info,
        )

        user_prompt = (
            f"{data_info_text}\n"
            f"以下是 {len(compact_predictions)} 个模型预测(JSON)：\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "请按规则输出 JSON。"
        )

        response_text = None
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            try:
                completion = self.client.chat.completions.create(
                    **kwargs,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception:
                completion = self.client.chat.completions.create(**kwargs)

            if not completion.choices:
                return self._fallback_single(predictions)

            message = completion.choices[0].message
            response_text = (message.content or "").strip() if message else ""
            if not response_text:
                return self._fallback_single(predictions)

            json_text = self._extract_json_from_text(response_text)
            decision_data = json.loads(json_text)

            selected_model = str(decision_data.get("selected_model", "")).strip()
            selected_class = str(decision_data.get("selected_class", "")).strip()
            confidence = float(decision_data.get("confidence", 0.0))
            reasoning = str(decision_data.get("reasoning", "")).strip()

            if not selected_model or not selected_class:
                return self._fallback_single(predictions)

            valid_names = {prediction.model_name for prediction in predictions}
            if selected_model not in valid_names:
                return self._fallback_single(predictions)

            return AgentDecision(
                selected_model=selected_model,
                selected_class=selected_class,
                confidence=float(confidence),
                reasoning=reasoning or "按多模型输出综合选择最可信结果。",
                all_predictions=[prediction.to_dict() for prediction in predictions],
            )
        except Exception:
            return self._fallback_single(predictions)

    def select_best_model_batch(
        self,
        batch_data: List[Dict[str, Any]],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
        incremental_save_path: Optional[str] = None,
    ) -> List[AgentDecision]:
        if not batch_data:
            raise ValueError("No batch data provided")

        decisions: List[AgentDecision] = []
        for item in batch_data:
            decision = self.select_best_model(
                predictions=item["predictions"],
                use_json_format=use_json_format,
                input_device_info=input_device_info,
                input_data_info=input_data_info,
            )
            decisions.append(decision)

            if incremental_save_path:
                self._save_incremental(decisions, batch_data[: len(decisions)], incremental_save_path)

        return decisions

    @staticmethod
    def _save_incremental(
        decisions: List[AgentDecision],
        data: List[Dict[str, Any]],
        path: str,
    ) -> None:
        results = []
        for item, decision in zip(data, decisions):
            preds = decision.all_predictions or []
            results.append(
                {
                    "image_file": item.get("image_file", ""),
                    "image_name": item.get("image_name", ""),
                    "selected_model": decision.selected_model,
                    "predicted_class": decision.selected_class,
                    "confidence": float(decision.confidence),
                    "reasoning": decision.reasoning,
                    "all_predictions": [
                        {
                            "model": pred.get("model_name", ""),
                            "top_class": pred.get("top_class", ""),
                            "top_confidence": float(pred.get("top_confidence", 0.0)),
                            "predictions": {
                                key: float(value)
                                for key, value in (pred.get("predictions") or {}).items()
                            },
                        }
                        for pred in preds
                    ],
                }
            )
        with open(path, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
