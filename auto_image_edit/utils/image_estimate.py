from typing import List
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import f1_score  # 新增依赖：pip install scikit-image
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pointbiserialr, spearmanr


class ImageEstimate:
    @staticmethod
    def mse_with_weight(origin_img: np.ndarray, edited_img: np.ndarray) -> float:
        """
        加权均方误差：冷区权重=1，热区权重=原始值
        提供整体误差的度量

        MSE设计用于回归任务，测量数值预测的准确性
        MSE值缺乏清晰的上下界, 难以直观解释

        公式：w_i = { 1 if y_i=0, y_i if y_i>0 }
        """
        # 创建权重矩阵：冷区=1，热区=原始值
        weight = np.where(origin_img == 0, 1, origin_img)

        sum_w = np.sum(weight)
        if sum_w == 0:
            return float("nan")

        # 计算加权均方误差
        squared_diff = (origin_img - edited_img) ** 2
        return float(np.sum(weight * squared_diff) / sum_w)

    @staticmethod
    def mape_with_weight(origin_img: np.ndarray, edited_img: np.ndarray) -> float:
        """
        加权平均绝对百分比误差：冷区权重=1，热区权重=原始值
        只能在正样本区域计算
        公式：w_i = { 1 if y_i=0, y_i if y_i>0 }
        """
        # 创建权重矩阵：冷区=1，热区=原始值
        weight = np.where(origin_img == 0, 1, origin_img)

        sum_w = np.sum(weight)
        if sum_w == 0:
            return float("nan")

        # 仅处理非零像素（避免除零错误）
        mask = origin_img > 0
        abs_err = np.abs(origin_img[mask] - edited_img[mask])
        rel_err = abs_err / origin_img[mask]

        return float(np.sum(weight[mask] * rel_err) / sum_w)

    @staticmethod
    def metric_psnr(origin_img: np.ndarray, edited_img: np.ndarray, max_val: float = None) -> float:
        """
        峰值信噪比（PSNR）
        PSNR = 10 * log10(max_val^2 / MSE)
        """
        mse = np.mean((origin_img - edited_img) ** 2)
        if mse == 0:
            return float("inf")
        if max_val is None:
            max_val = origin_img.max()
        return float(10 * np.log10((max_val**2) / mse))

    @staticmethod
    def metric_ssim(origin_img: np.ndarray, edited_img: np.ndarray, data_range: float = None) -> float:
        """
        结构相似性指数（SSIM）
        """
        if data_range is None:
            data_range = origin_img.max() - origin_img.min()
        return float(structural_similarity(origin_img, edited_img, data_range=data_range))

    @staticmethod
    def metric_r_square(origin_img: np.ndarray, edited_img: np.ndarray) -> float:
        """
        R-squared (Coefficient of Determination)
        R^2 = 1 - (SS_res / SS_tot) = SSR / SST
        其中 SS_res 是残差平方和，SS_tot 是总平方和
        SSR = np.sum((edited_img - np.mean(edited_img)) ** 2)
        SST = np.sum((origin_img - np.mean(origin_img)) ** 2)
        """
        # 计算原始图像的均值
        origin_mean = np.mean(origin_img)
        # 残差平方和
        ss_res = np.sum((origin_img - edited_img) ** 2)
        # 总平方和
        ss_tot = np.sum((origin_img - origin_mean) ** 2)
        # 若原始图像恒定，返回 NaN
        if ss_tot == 0:
            return float("nan")
        # 计算并返回 R²
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def metric_ap(
        origin: np.ndarray,
        edited: np.ndarray,
        start_threshold: float,
        end_threshold: float,
        step: float = 0.1,
    ) -> float:
        """
        平均精度（AP）
        1. origin 和 edited 的取值范围为 [0,1]
        2. 对一系列阈值计算 TP/FP/FN，再算 precision 和 recall
        3. 补齐端点、做单调插值后梯形积分
        """
        if origin.shape != edited.shape:
            raise ValueError("Origin and edited images must have the same shape.")

        # 生成阈值序列，上限微调避免浮点误差
        thresholds = np.arange(start_threshold, end_threshold + step / 2, step)

        # 逐阈值计算 TP, FP, FN，降低内存占用峰值
        tp_list, fp_list, fn_list = [], [], []
        for thr in thresholds:
            bin_o = origin > thr
            bin_e = edited > thr
            tp_list.append(np.logical_and(bin_o, bin_e).sum())
            fp_list.append(np.logical_and(~bin_o, bin_e).sum())
            fn_list.append(np.logical_and(bin_o, ~bin_e).sum())
        tp = np.array(tp_list)
        fp = np.array(fp_list)
        fn = np.array(fn_list)

        # 计算 precision 和 recall，并屏蔽除零警告
        with np.errstate(divide="ignore", invalid="ignore"):
            precisions = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            recalls = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

        # 补齐 (recall,precision) 端点，并做单调非增插值
        prec = np.concatenate(([1.0], precisions, [0.0]))
        rec = np.concatenate(([0.0], recalls, [1.0]))
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])

        # 按 recall 升序积分
        order = np.argsort(rec)
        ap = np.trapezoid(prec[order], x=rec[order])
        return float(ap)

    @staticmethod
    def metric_f1(
        origin: np.ndarray,
        edited: np.ndarray,
        start_threshold: float,
        end_threshold: float,
        step: float = 0.1,
        average: str = "binary",
    ) -> np.ndarray:
        """
        计算各阈值下的 F1
        """
        if origin.shape != edited.shape:
            raise ValueError("Origin and edited images must have the same shape.")

        thresholds = np.arange(start_threshold, end_threshold + step / 2, step)
        f1_list = []
        for thr in thresholds:
            bin_o = (origin > thr).ravel()
            bin_e = (edited > thr).ravel()
            f1 = f1_score(bin_o, bin_e, average=average)
            f1_list.append(f1)
        return np.mean(f1_list)

    @staticmethod
    def metric_iou(
        origin: np.ndarray,
        edited: np.ndarray,
        start_threshold: float,
        end_threshold: float,
        step: float = 0.1,
    ) -> np.ndarray:
        """
        计算各阈值下的 IoU
        """
        if origin.shape != edited.shape:
            raise ValueError("Origin and edited images must have the same shape.")

        thresholds = np.arange(start_threshold, end_threshold + step / 2, step)
        iou_list = []
        for thr in thresholds:
            bin_o = (origin > thr).ravel()
            bin_e = (edited > thr).ravel()
            intersection = np.logical_and(bin_o, bin_e).sum()
            union = np.logical_or(bin_o, bin_e).sum()
            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)
        return np.mean(iou_list)

    @staticmethod
    def metric_pr_auc(origin: np.ndarray, edited: np.ndarray) -> float:
        """
        计算各阈值下的 AUC_PR和AUC_ROC
        """
        # auc_roc = roc_auc_score(origin, edited)
        pr_score = average_precision_score(origin, edited)
        return pr_score
        # return {"auc_roc": auc_roc, "pr_score": pr_score}

    @staticmethod
    def metric_correlation(
        origin: np.ndarray,
        edited: np.ndarray,
    ) -> dict:
        """
        计算点二列相关系数和斯皮尔曼相关系数
        """
        # 确保输入是一维数组
        origin_flat = origin.flatten()
        edited_flat = edited.flatten()

        # 斯皮尔曼相关系数
        spearmanr_corr, spearman_p_value = spearmanr(origin_flat, edited_flat)
        return spearmanr_corr

    @staticmethod
    def metric_soft_iou(origin: np.ndarray, edited: np.ndarray, epsilon=1e-6) -> np.ndarray:
        """
        计算各阈值下的 Soft IoU
        """
        if origin.shape != edited.shape:
            raise ValueError("Origin and edited images must have the same shape.")

        # 确保输入是浮点类型
        edited = edited.astype(np.float32)
        origin = origin.astype(np.float32)
        # 软交集 (Soft Intersection)
        # 逐元素相乘，然后求和
        intersection = np.sum(edited * origin)
        # 软并集 (Soft Union)
        # 公式: P(A union B) = P(A) + P(B) - P(A intersection B)
        # 同样逐元素计算后求和
        union = np.sum(edited + origin - edited * origin)
        # 计算 Soft IoU
        # 加上 epsilon 防止分母为零
        iou = intersection / (union + epsilon)
        return iou

    @staticmethod
    def metric(gt_mask, pred_mask):
        # 确保输入是一维数组
        gt_mask_flat = gt_mask.flatten()
        pred_mask_flat = pred_mask.flatten()

        # 检查是否为常数数组
        if len(np.unique(gt_mask_flat)) == 1 or len(np.unique(pred_mask_flat)) == 1:
            raise ValueError("输入数组不能是常数数组")
        ret = {}
        ret["pr_auc"] = ImageEstimate.metric_pr_auc(gt_mask_flat, pred_mask_flat)
        ret["wmse"] = ImageEstimate.mse_with_weight(gt_mask_flat, pred_mask_flat)
        ret["soft_iou"] = ImageEstimate.metric_soft_iou(gt_mask_flat, pred_mask_flat)
        ret["r_square"] = ImageEstimate.metric_r_square(gt_mask_flat, pred_mask_flat)
        ret["ssim"] = ImageEstimate.metric_ssim(gt_mask_flat, pred_mask_flat)
        return ret
