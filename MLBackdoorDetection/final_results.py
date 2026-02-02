import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import bisect

def compute_scores(series, center=None, mad=None, tail="upper"):
    """
    Robust one-sided score w.r.t. benign-train distribution:
    - tail='upper': score = max(0, (x-center)/MAD)
    - tail='lower': score = max(0, (center-x)/MAD)
    """
    if center is None:
        center = np.median(series)
    if mad is None:
        mad = np.median(np.abs(series - center))
    if mad == 0:
        mad = 1e-6
    z = (series - center) / mad
    if tail == "upper":
        scores = np.maximum(0.0, z)
    else:
        scores = np.maximum(0.0, -z)
    return scores, float(center), float(mad)

def pick_threshold_min_fpr(scores_benign_train, max_fpr=0.05):
    # 在良性训练分数上选择“最小阈值”使 FPR ≤ max_fpr
    s = np.sort(np.asarray(scores_benign_train))
    if len(s) == 0:
        return float("inf")
    idx = int(np.ceil((1 - max_fpr) * len(s))) - 1
    idx = np.clip(idx, 0, len(s) - 1)
    return float(s[idx])

def pick_threshold_max_f1_under_fpr(ben_scores_train, tro_scores_train, max_fpr=0.05):
    """
    在校准集上搜索阈值，使良性 FPR≤max_fpr 且 F1 最大。
    """
    b = np.asarray(ben_scores_train)
    t = np.asarray(tro_scores_train)
    if len(b) == 0 or len(t) == 0:
        return pick_threshold_min_fpr(b, max_fpr)

    cand = np.unique(np.concatenate([b, t]))
    best_thr, best_f1 = None, -1.0
    b_sorted = np.sort(b)
    nb = len(b_sorted)

    def fpr_at_thr(thr):
        idx_right = bisect.bisect_right(b_sorted, thr)
        fp = nb - idx_right
        return fp / nb if nb > 0 else 0.0

    for thr in cand:
        if fpr_at_thr(thr) <= max_fpr:
            y_true = np.concatenate([np.zeros_like(b, dtype=int), np.ones_like(t, dtype=int)])
            y_pred = np.concatenate([(b > thr).astype(int), (t > thr).astype(int)])
            f1 = f1_score(y_true, y_pred) if y_pred.sum() > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

    if best_thr is None:
        best_thr = pick_threshold_min_fpr(b, max_fpr)
    return best_thr

def choose_tail_by_calib(ben_train_series, tro_train_series, center, mad):
    """
    在校准集上选择 tail（upper/lower）以最大化 AUROC。
    """
    z_b = (np.asarray(ben_train_series) - center) / (mad if mad != 0 else 1e-6)
    z_t = (np.asarray(tro_train_series) - center) / (mad if mad != 0 else 1e-6)
    y_true = np.concatenate([np.zeros_like(z_b, dtype=int), np.ones_like(z_t, dtype=int)])
    # upper: 分数=z；lower: 分数=-z
    au_upper = roc_auc_score(y_true, np.concatenate([z_b, z_t])) if len(z_b) and len(z_t) else 0.5
    au_lower = roc_auc_score(y_true, np.concatenate([-z_b, -z_t])) if len(z_b) and len(z_t) else 0.5
    return "upper" if au_upper >= au_lower else "lower"

def evaluate_once(df_benign, df_trojan, train_ratio=0.3, max_fpr=0.05, rng=None, thr_policy="min_fpr"):
    if rng is None:
        rng = np.random.default_rng()

    # 随机划分（模型级）——1:1 校准集
    n_b = len(df_benign)
    n_t = len(df_trojan)
    idx_b = rng.permutation(n_b)
    idx_t = rng.permutation(n_t)
    n_b_target = max(1, int(train_ratio * n_b))
    n_t_target = max(1, int(train_ratio * n_t))
    n_train_each = max(1, min(n_b_target, n_t_target))

    ben_train = df_benign.iloc[idx_b[:n_train_each]].copy()
    tro_train = df_trojan.iloc[idx_t[:n_train_each]].copy()
    ben_test  = df_benign.iloc[idx_b[n_train_each:]].copy()
    tro_test  = df_trojan.iloc[idx_t[n_train_each:]].copy()

    # 用良性训练集拟合 center/mad
    _, center, mad = compute_scores(ben_train['anomaly metric'])
    # 在校准集上选择 tail 以最大化 AUROC
    tail = choose_tail_by_calib(ben_train['anomaly metric'], tro_train['anomaly metric'], center, mad)

    # 校准分数（用于选阈值）
    ben_train_scores, _, _ = compute_scores(ben_train['anomaly metric'], center=center, mad=mad, tail=tail)
    tro_train_scores, _, _ = compute_scores(tro_train['anomaly metric'], center=center, mad=mad, tail=tail)

    if thr_policy == "max_f1":
        thr = pick_threshold_max_f1_under_fpr(ben_train_scores, tro_train_scores, max_fpr=max_fpr)
    else:
        thr = pick_threshold_min_fpr(ben_train_scores, max_fpr=max_fpr)

    # 测试集分数（统一良性训练的 center/mad 与选定 tail）
    ben_test_scores, _, _ = compute_scores(ben_test['anomaly metric'], center=center, mad=mad, tail=tail)
    tro_test_scores, _, _ = compute_scores(tro_test['anomaly metric'], center=center, mad=mad, tail=tail)

    y_true = np.concatenate([np.zeros(len(ben_test_scores)), np.ones(len(tro_test_scores))])
    y_scores = np.concatenate([ben_test_scores, tro_test_scores])
    y_pred = (y_scores > thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res = {
        "F1": f1_score(y_true, y_pred) if (tp+fp)>0 and (tp+fn)>0 else 0.0,
        "AUROC": roc_auc_score(y_true, y_scores) if len(np.unique(y_true))>1 else 0.5,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall (TPR)": recall_score(y_true, y_pred) if (tp+fn)>0 else 0.0,
        "Precision": precision_score(y_true, y_pred) if (tp+fp)>0 else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "Threshold": float(thr),
        "Median": float(center),
        "MAD": float(mad),
        "Tail": tail,
        "Benign_train": len(ben_train),
        "Benign_test": len(ben_test),
        "Trojan_train": len(tro_train),
        "Trojan_test": len(tro_test),
    }
    return res

def main():
    parser = argparse.ArgumentParser(description="FREEEAGLE-style evaluation with FPR-constrained threshold.")
    parser.add_argument("--benign_csv", required=True, help="CSV for benign models (columns: filepath, anomaly metric, time cost)")
    parser.add_argument("--trojan_csv", required=True, help="CSV for trojaned models (columns: filepath, anomaly metric, time cost)")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeats")
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Train split ratio")
    parser.add_argument("--max_fpr", type=float, default=0.05, help="FPR constraint")
    parser.add_argument("--thr_policy", choices=["min_fpr","max_f1"], default="min_fpr",
                        help="Threshold policy under FPR constraint (min_fpr or maximize F1 on calibration set)")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed for reproducible splits (default: 42)")
    args = parser.parse_args()

    df_benign = pd.read_csv(args.benign_csv)
    df_trojan = pd.read_csv(args.trojan_csv)
    for df in (df_benign, df_trojan):
        if "anomaly metric" not in df.columns:
            raise ValueError("CSV must contain 'anomaly metric' column")

    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(args.repeat):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        res = evaluate_once(
            df_benign, df_trojan,
            train_ratio=args.train_ratio, max_fpr=args.max_fpr, rng=local_rng,
            thr_policy=args.thr_policy
        )
        results.append(res)

    df_res = pd.DataFrame(results)
    print("\n--- Averaged over {} repeats ---".format(args.repeat))
    print(f"Benign CSV: {args.benign_csv}")
    print(f"Trojan CSV: {args.trojan_csv}")
    for k in ["F1", "AUROC", "Accuracy", "Recall (TPR)", "Precision", "FPR", "Threshold", "Median", "MAD"]:
        print(f"{k}: {df_res[k].mean():.4f} ± {df_res[k].std():.4f}")

    # 新增：直接给出 F1/AUROC 的最佳结果（max）
    best_f1_idx = int(df_res["F1"].idxmax()) if len(df_res) else -1
    best_auc_idx = int(df_res["AUROC"].idxmax()) if len(df_res) else -1
    if best_f1_idx != -1:
        print(f"Best F1: {df_res.at[best_f1_idx, 'F1']:.4f} at repeat {best_f1_idx}")
    if best_auc_idx != -1:
        print(f"Best AUROC: {df_res.at[best_auc_idx, 'AUROC']:.4f} at repeat {best_auc_idx}")

if __name__ == "__main__":
    main()