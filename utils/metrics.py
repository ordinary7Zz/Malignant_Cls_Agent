import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from tqdm import tqdm
from typing import Optional, List
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    roc_curve
)

# =========================
# Dice（逐病例）
# =========================
class Dice(nn.Module):
    """
    Dice coefficient calculator for binary segmentation tasks.
    """
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, predict, target):
        smooth = 1.0
        intersection = (predict * target).sum()
        dice = (2.0 * intersection + smooth) / (
            predict.sum() + target.sum() + smooth
        )
        return dice


# =========================
# HD95（逐病例）
# =========================
class HD95(nn.Module):
    """
    HD95 calculator for binary segmentation tasks.
    使用距离变换方法计算 Hausdorff Distance (95%)
    """
    def __init__(self):
        super(HD95, self).__init__()

    def forward(self, predict, target):
        return self.calculate_hd(predict, target)

    def calculate_hd(self, predict, target):
        # 处理空 mask（工程可跑，但论文中需说明）
        if predict.sum() == 0:
            predict = predict.clone()
            predict[0, 0, 0] = 1.0
        if target.sum() == 0:
            target = target.clone()
            target[0, 0, 0] = 1.0

        pred_np = predict.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)

        hd1 = self.hd_distance(pred_np, target_np)
        hd2 = self.hd_distance(target_np, pred_np)

        return torch.tensor(max(hd1, hd2), dtype=torch.float32)

    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        indexes = np.nonzero(x)
        distances = edt(~y)
        return float(np.percentile(distances[indexes], 95))


# =========================
# Bootstrap CI95（通用）
# =========================
def bootstrap_ci(values, n_boot=5000, ci=0.95, seed=0):
    """
    基于逐病例指标计算 mean 和 CI95（bootstrap）
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return 0.0, (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1.0 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return values.mean(), (lower, upper)


def _safe_roc_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def classification_bootstrap_metrics(y_probs, y_labels, threshold=0.5, n_boot=2000, ci=0.95, seed=0):
    """
    使用 bootstrap 在样本级别估计二分类指标的均值及 CI95。
    返回一个 dict, 每个指标包含 (mean, (lower, upper))
    指标: accuracy, precision, recall, f1, auroc, auprc, sensitivity, specificity, youden
    """
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_labels = np.asarray(y_labels, dtype=np.int32)

    # 过滤无效标签（-1 表示缺失）
    valid_mask = (y_labels != -1)
    y_probs = y_probs[valid_mask]
    y_labels = y_labels[valid_mask]

    if y_labels.size == 0:
        zero_ci = (0.0, (0.0, 0.0))
        return {k: zero_ci for k in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc', 'sensitivity', 'specificity', 'youden']}

    rng = np.random.default_rng(seed)
    n = y_labels.size

    metrics_samples = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'auroc': [], 'auprc': [], 'sensitivity': [], 'specificity': [], 'youden': []
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_probs[idx]
        labels_s = y_labels[idx]

        preds_s = (probs_s >= float(threshold)).astype(int)

        tp = np.sum((preds_s == 1) & (labels_s == 1))
        tn = np.sum((preds_s == 0) & (labels_s == 0))
        fp = np.sum((preds_s == 1) & (labels_s == 0))
        fn = np.sum((preds_s == 0) & (labels_s == 1))

        acc = float((preds_s == labels_s).mean())
        prec = precision_score(labels_s, preds_s, zero_division=0)
        rec = recall_score(labels_s, preds_s, zero_division=0)
        f1 = f1_score(labels_s, preds_s, zero_division=0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0

        try:
            auroc = roc_auc_score(labels_s, probs_s) if len(np.unique(labels_s)) > 1 else float('nan')
        except Exception:
            auroc = float('nan')
        try:
            auprc = average_precision_score(labels_s, probs_s) if len(np.unique(labels_s)) > 1 else float('nan')
        except Exception:
            auprc = float('nan')

        metrics_samples['accuracy'].append(acc)
        metrics_samples['precision'].append(float(prec))
        metrics_samples['recall'].append(float(rec))
        metrics_samples['f1'].append(float(f1))
        metrics_samples['sensitivity'].append(float(sens))
        metrics_samples['specificity'].append(float(spec))
        metrics_samples['youden'].append(float(youden))
        metrics_samples['auroc'].append(float(np.nan if np.isnan(auroc) else auroc))
        metrics_samples['auprc'].append(float(np.nan if np.isnan(auprc) else auprc))

    results = {}
    alpha = 1.0 - ci
    for k, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float32)
        # 过滤 nan（例如在 bootstrap sample 中某一类缺失导致 AUROC nan）
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[k] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[k] = (mean, (lower, upper))

    return results


def multiclass_bootstrap_metrics(probs_list, labels_list, n_boot=2000, ci=0.95, seed=0):
    """
    对多分类（TIRADS）使用 bootstrap 估计 accuracy/precision/recall/f1/auc 的均值与 CI。
    probs_list: list 或 np.array, 每项为预测的概率向量（长度 = num_classes）
    labels_list: list 或 np.array 的整数标签
    """
    probs = np.asarray(probs_list, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int32)

    if labels.size == 0:
        zero_ci = (0.0, (0.0, 0.0))
        return {k: zero_ci for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}

    rng = np.random.default_rng(seed)
    n = labels.size

    metrics_samples = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = probs[idx]
        labels_s = labels[idx]

        preds_s = np.argmax(probs_s, axis=1)

        acc = float((preds_s == labels_s).mean())
        prec = precision_score(labels_s, preds_s, average='macro', zero_division=0)
        rec = recall_score(labels_s, preds_s, average='macro', zero_division=0)
        f1 = f1_score(labels_s, preds_s, average='macro', zero_division=0)

        # AUC for multiclass: require >1 class
        try:
            unique = np.unique(labels_s)
            if unique.size > 1:
                auc = roc_auc_score(labels_s, probs_s, multi_class='ovr', average='macro')
            else:
                auc = float('nan')
        except Exception:
            auc = float('nan')

        metrics_samples['accuracy'].append(acc)
        metrics_samples['precision'].append(float(prec))
        metrics_samples['recall'].append(float(rec))
        metrics_samples['f1'].append(float(f1))
        metrics_samples['auc'].append(float(np.nan if np.isnan(auc) else auc))

    results = {}
    alpha = 1.0 - ci
    for k, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float32)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[k] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[k] = (mean, (lower, upper))

    return results


# =========================
# 模型评估（逐病例 + CI95）
# =========================
def evaluate_model(net, dataloader, device, threshold=0.5):
    """
    Evaluate segmentation model with:
    - Dice (per-case)
    - HD95 (per-case)
    - CI95 for both metrics (bootstrap)

    Returns:
        dict with mean + CI95
    """
    net.eval()

    dice_calculator = Dice()
    hd_calculator = HD95()

    all_dice_values = []
    all_hd_values = []

    # For classification
    all_malignancy_probs = []
    all_malignancy_labels = []

    all_tirads_probs = []
    all_tirads_labels = []

    for batch in tqdm(dataloader, desc="Evaluating Model", leave=False):
        # 兼容不同 batch 格式，并收集分类标签/概率
        if isinstance(batch, dict):
            image = batch["image"]
            mask_true = batch["label"]
            malignancy_labels = batch.get('malignancy', None)
            tirads_labels = batch.get('tirads', None)
        else:
            # 支持 (image, label, malignancy, tirads)
            if len(batch) >= 4:
                image, mask_true, malignancy_labels, tirads_labels = batch[0], batch[1], batch[2], batch[3]
            else:
                image, mask_true = batch[0], batch[1]
                malignancy_labels, tirads_labels = None, None

        image = image.to(device)
        mask_true = mask_true.to(device)

        with torch.no_grad():
            outputs = net(image)

            # 支持多输出 (mask, malignancy_logits, tirads_logits)
            if isinstance(outputs, (list, tuple)):
                mask_pred = outputs[0]

                # collect malignancy
                if len(outputs) > 1 and malignancy_labels is not None:
                    malignancy_logits = outputs[1]
                    malignancy_labels = malignancy_labels.to(device)
                    # valid mask
                    valid_m = (malignancy_labels != -1)
                    if valid_m.any():
                        mal_pred = malignancy_logits[valid_m]
                        if mal_pred.dim() == 1:
                            mal_pred = mal_pred.unsqueeze(1)
                        mal_probs = torch.sigmoid(mal_pred).squeeze(1)
                        all_malignancy_probs.extend(mal_probs.detach().cpu().numpy().tolist())
                        all_malignancy_labels.extend(malignancy_labels[valid_m].cpu().numpy().tolist())

                # collect tirads
                if len(outputs) > 2 and tirads_labels is not None:
                    tirads_logits = outputs[2]
                    tirads_labels = tirads_labels.to(device)
                    valid_t = (tirads_labels != -1)
                    if valid_t.any():
                        tir_pred = tirads_logits[valid_t]
                        tir_probs = torch.softmax(tir_pred, dim=1)
                        all_tirads_probs.extend(tir_probs.detach().cpu().numpy().tolist())
                        all_tirads_labels.extend(tirads_labels[valid_t].cpu().numpy().tolist())
            else:
                mask_pred = outputs

            mask_pred = torch.sigmoid(mask_pred)
            mask_pred_binary = (mask_pred > 0.5).float()

        batch_size = image.size(0)

        # 逐病例计算 Dice / HD95
        for i in range(batch_size):
            pred_i = mask_pred_binary[i]
            true_i = (mask_true[i] > 0.5).float()

            # Dice
            dice_i = dice_calculator(pred_i, true_i).item()
            all_dice_values.append(dice_i)

            # HD95
            try:
                hd_i = hd_calculator(pred_i, true_i).item()
                all_hd_values.append(hd_i)
            except Exception as e:
                print(f"[Warning] HD95 failed on sample {i}: {e}")

    net.train()

    # =========================
    # 计算 mean + CI95
    # =========================
    dice_mean, dice_ci95 = bootstrap_ci(all_dice_values)
    hd95_mean, hd95_ci95 = bootstrap_ci(all_hd_values)

    # classification bootstrap
    malignancy_metrics_ci = classification_bootstrap_metrics(all_malignancy_probs, all_malignancy_labels, threshold=threshold)
    tirads_metrics_ci = multiclass_bootstrap_metrics(all_tirads_probs, all_tirads_labels)

    rounded_dice_values = [round(float(v), 4) for v in all_dice_values]
    rounded_hd_values = [round(float(v), 4) for v in all_hd_values]

    results = {
        "Dice": {
            "mean": round(dice_mean, 4),
            "CI95": (round(dice_ci95[0], 4), round(dice_ci95[1], 4)),
            "values": rounded_dice_values,
        },
        "HD95": {
            "mean": round(hd95_mean, 4),
            "CI95": (round(hd95_ci95[0], 4), round(hd95_ci95[1], 4)),
            "values": rounded_hd_values,
        },
        "Malignancy": {},
        "TIRADS": {},
    }

    # 填充良/恶性结果（包含均值与 CI）
    for k, v in malignancy_metrics_ci.items():
        mean_v, (low_v, high_v) = v
        results['Malignancy'][k] = {
            'mean': round(mean_v, 4),
            'CI95': (round(low_v, 4), round(high_v, 4)),
        }

    # 填充 TIRADS 结果
    for k, v in tirads_metrics_ci.items():
        mean_v, (low_v, high_v) = v
        results['TIRADS'][k] = {
            'mean': round(mean_v, 4),
            'CI95': (round(low_v, 4), round(high_v, 4)),
        }

    return results

def find_best_threshold_by_youden_index(
    y_true,
    y_prob,
    thresholds: Optional[np.ndarray] = None,
):
    """Find the best binary classification threshold by maximizing Youden Index.

    Youden Index: J = sensitivity + specificity - 1

    Args:
        y_true: array-like of shape (N,), labels in {0,1}. Values -1 will be ignored.
        y_prob: array-like of shape (N,), predicted probabilities for class 1.
        thresholds: optional np.ndarray of thresholds to search.
            - If provided: use these thresholds (grid search).
            - If None: use sklearn.metrics.roc_curve-derived thresholds (recommended).

    Returns:
        dict with keys: best_threshold, youden, sensitivity, specificity
    """
    y_true_np = np.asarray(y_true).reshape(-1)
    y_prob_np = np.asarray(y_prob).reshape(-1)

    valid_mask = y_true_np != -1
    y_true_np = y_true_np[valid_mask]
    y_prob_np = y_prob_np[valid_mask]

    if y_true_np.size == 0:
        return {
            'best_threshold': 0.5,
            'youden': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
        }

    unique_labels = np.unique(y_true_np)
    if unique_labels.size < 2:
        # Youden is undefined if only one class exists
        return {
            'best_threshold': 0.5,
            'youden': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
        }

    best_t = 0.5
    best_j = -np.inf
    best_sens = 0.0
    best_spec = 0.0

    if thresholds is None:
        # roc_curve gives thresholds at which predictions change; maximizing Youden on these is exact
        # for the dataset (no need for 0..1 grid).
        fpr, tpr, roc_thresholds = roc_curve(y_true_np, y_prob_np, pos_label=1)
        youden = tpr - fpr
        max_j = np.max(youden)
        candidate_idx = np.where(youden == max_j)[0]

        # Tie-break: prefer higher sensitivity, then lower threshold
        if candidate_idx.size > 1:
            best_tpr = np.max(tpr[candidate_idx])
            candidate_idx = candidate_idx[np.where(tpr[candidate_idx] == best_tpr)[0]]

        idx = int(candidate_idx[np.argmin(roc_thresholds[candidate_idx])])
        best_t = float(roc_thresholds[idx])
        best_j = float(youden[idx])
        best_sens = float(tpr[idx])
        best_spec = float(1.0 - fpr[idx])
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
        if thresholds.size == 0:
            thresholds = np.array([0.5], dtype=np.float32)

        best_t = float(thresholds[0])
        for t in thresholds:
            # Use ">=" to align with sklearn's ROC thresholding convention.
            preds = (y_prob_np >= float(t)).astype(int)

            tp = np.sum((preds == 1) & (y_true_np == 1))
            tn = np.sum((preds == 0) & (y_true_np == 0))
            fp = np.sum((preds == 1) & (y_true_np == 0))
            fn = np.sum((preds == 0) & (y_true_np == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sensitivity + specificity - 1.0

            # Tie-break: prefer higher J, then higher sensitivity, then lower threshold
            if (j > best_j) or (j == best_j and sensitivity > best_sens) or (j == best_j and sensitivity == best_sens and float(t) < best_t):
                best_j = float(j)
                best_t = float(t)
                best_sens = float(sensitivity)
                best_spec = float(specificity)

    return {
        'best_threshold': round(best_t, 6),
        'youden': round(best_j, 6),
        'sensitivity': round(best_sens, 6),
        'specificity': round(best_spec, 6),
    }


def compute_youden_threshold(
    net,
    dataloader,
    device,
    thresholds: Optional[np.ndarray] = None,
):
    """Compute the best malignancy threshold on a validation dataloader via Youden index.

    This function ONLY computes the threshold; it does NOT evaluate test performance.
    Compute it once on the validation set and reuse it across multiple test datasets.

    Returns:
      threshold_info dict with keys: best_threshold, youden, sensitivity, specificity
    """
    net.eval()

    all_malignancy_probs: List[float] = []
    all_malignancy_labels: List[int] = []

    try:
        num_batches = len(dataloader)
    except Exception:
        num_batches = None

    for batch in tqdm(
        dataloader,
        total=num_batches,
        desc='Collecting Malignancy Probs (Youden)',
        unit='batch',
        leave=False,
    ):
        try:
            if isinstance(batch, dict):
                image = batch['image']
                malignancy_labels = batch['malignancy']
            else:
                image = batch[0]
                malignancy_labels = batch[2]

            image = image.to(device=device)
            malignancy_labels = malignancy_labels.to(device=device)

            with torch.no_grad():
                outputs = net(image)

                if not isinstance(outputs, (list, tuple)):
                    continue

                malignancy_pred = outputs[1]

                valid_mal_mask = (malignancy_labels != -1)
                if not valid_mal_mask.any():
                    continue

                valid_mal_pred = malignancy_pred[valid_mal_mask]
                valid_mal_labels = malignancy_labels[valid_mal_mask]

                if valid_mal_pred.dim() == 1:
                    valid_mal_pred = valid_mal_pred.unsqueeze(1)
                mal_probs = torch.sigmoid(valid_mal_pred).squeeze(1)

                all_malignancy_probs.extend(mal_probs.detach().cpu().numpy().tolist())
                all_malignancy_labels.extend(valid_mal_labels.detach().cpu().numpy().tolist())
        except Exception as e:
            print(f"Error collecting malignancy probs: {e}")
            continue

    net.train()

    threshold_info = find_best_threshold_by_youden_index(
        all_malignancy_labels,
        all_malignancy_probs,
        thresholds=thresholds,
    )

    return threshold_info