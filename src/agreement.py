# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

"""
agreement.py —— 多评分者信度与最终相关性
=========================================
输入（自动探测，优先 CSV）：
    评分者A = misreading_scored_round2.csv 的 human_score
    评分者B = misreading_scored_round3.csv，无则 人工评分表_评分者B.xlsx
    评分者C = misreading_scored_round4.csv，无则 人工评分表_评分者C.xlsx
输出：
    1. 两两一致性（Pearson / Spearman / 二次加权 Kappa / 完全一致率）
    2. 整体信度（ICC(2,1)、ICC(2,k)、Kendall's W）
    3. 多种基准口径下的模型相关性
"""
import os
import sys
import warnings
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

B = os.path.dirname(os.path.abspath(__file__))
LABSE_CSV = os.path.join(DATA, "labse_results.csv")

RATERS = [
    ("A", os.path.join(DATA, "misreading_scored_round2.csv"), os.path.join(DATA, "rater_A.xlsx")),
    ("B", os.path.join(DATA, "misreading_scored_round3.csv"), os.path.join(DATA, "rater_B.xlsx")),
    ("C", os.path.join(DATA, "misreading_scored_round4.csv"), os.path.join(DATA, "rater_C.xlsx")),
]
# 独立评分者（非作者），用于排除作者偏好的敏感性分析
INDEPENDENT = ["B", "C"]


def load_one(csv_path, xlsx_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.set_index("pair_id")["human_score"].astype(float), os.path.basename(csv_path)
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path, sheet_name="评分表")
        except Exception as e:
            print(f"  ⚠️ 读取 {os.path.basename(xlsx_path)} 失败：{e}")
            return None, None
        col = next((c for c in df.columns if "语义一致度" in str(c)), None)
        if col is None:
            return None, None
        s = pd.to_numeric(df.set_index("编号")[col], errors="coerce").dropna().astype(float)
        return s, os.path.basename(xlsx_path)
    return None, None


def icc(M):
    """Shrout & Fleiss ICC(2,1)、ICC(2,k)：双向随机效应、绝对一致性。"""
    k = M.shape[1]; n = M.shape[0]
    X = M.values.astype(float)
    grand = X.mean()
    ss_rows = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((X - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1); ms_cols = ss_cols / (k - 1); ms_err = ss_err / ((n - 1) * (k - 1))
    icc21 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    icc2k = (ms_rows - ms_err) / (ms_rows + (ms_cols - ms_err) / n)
    return icc21, icc2k


def kendall_w(M):
    m, n = M.shape[1], M.shape[0]
    R = M.rank(axis=0).sum(axis=1).values
    S = ((R - R.mean()) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    return W, chi2, stats.chi2.sf(chi2, n - 1), n - 1


def wkappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a.astype(int), b.astype(int), weights="quadratic")


def boot_ci(x, y, n=10000, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(len(x)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[s])) < 2 or len(np.unique(y[s])) < 2:
            continue
        out.append(stats.spearmanr(x[s], y[s]).statistic)
    return np.percentile(out, [2.5, 97.5])


def perm_p(x, y, n=10000, seed=42):
    obs = abs(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        c += abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= obs
    return (c + 1) / (n + 1)


def main():
    print("=" * 78)
    print("多评分者信度与最终相关性")
    print("=" * 78)

    base = pd.read_csv(os.path.join(DATA, "misreading_scored_round2.csv"))
    n_total = len(base)
    raters, sources = {}, {}
    for name, csv_p, xlsx_p in RATERS:
        s, src = load_one(csv_p, xlsx_p)
        if s is None:
            continue
        if len(s) < n_total:
            print(f"  ↷ 评分者{name} 仅填 {len(s)}/{n_total}，跳过")
            continue
        raters[name], sources[name] = s, src

    names = list(raters.keys())
    print(f"参与的评分者：{names}")
    for k in names:
        print(f"  评分者{k} ← {sources[k]}  （分布 {raters[k].value_counts().sort_index().to_dict()}）")
    if len(names) < 2:
        print("\n不足 2 位评分者，无法计算评分者间信度。")
        return

    ids = sorted(set.intersection(*[set(s.index) for s in raters.values()]))
    M = pd.DataFrame({k: raters[k].reindex(ids) for k in names}).dropna()
    print(f"\n共同评分文本对：{len(M)} 组 (n={len(M)})")

    print("\n" + "-" * 78)
    print("一、两两一致性")
    print("-" * 78)
    print(f"{'对':<8}{'Pearson':>10}{'Spearman':>10}{'加权Kappa':>12}{'完全一致':>12}{'平均绝对差':>12}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = M[names[i]].values, M[names[j]].values
            same = int((a == b).sum())
            print(f"{names[i]}-{names[j]:<6}{stats.pearsonr(a, b).statistic:>10.4f}"
                  f"{stats.spearmanr(a, b).statistic:>10.4f}{wkappa(a, b):>12.4f}"
                  f"{same:>8}/{len(a):<4}{np.abs(a - b).mean():>12.4f}")

    print("\n" + "-" * 78)
    print("二、整体信度")
    print("-" * 78)
    i21, i2k = icc(M)
    print(f"  ICC(2,1) = {i21:.4f}   单评分者绝对一致性")
    print(f"  ICC(2,k) = {i2k:.4f}   k={len(names)} 位评分者均值的信度")
    W, chi2, p, df = kendall_w(M)
    print(f"  Kendall's W = {W:.4f}   χ²={chi2:.3f}, df={df}, p={p:.4f}")

    ind = [k for k in names if k in INDEPENDENT]
    if len(names) >= 3 and len(ind) >= 2:
        i21b, i2kb = icc(M[ind])
        Wb, _, pb, _ = kendall_w(M[ind])
        print(f"\n  仅独立评分者({''.join(ind)})：ICC(2,1)={i21b:.4f}，ICC(2,k)={i2kb:.4f}，W={Wb:.4f}(p={pb:.4f})")

    print("\n" + "-" * 78)
    print("三、不同基准口径下的模型相关性")
    print("-" * 78)
    lb = pd.read_csv(LABSE_CSV).set_index("pair_id")
    base_idx = base.set_index("pair_id")

    criteria = [("全部%d人均值" % len(names), M.mean(axis=1))]
    if len(ind) >= 2:
        criteria.append(("仅独立%d人均值" % len(ind), M[ind].mean(axis=1)))
    for k in names:
        criteria.append((f"仅评分者{k}", M[k]))

    print(f"{'基准':<18}{'n':>4}{'方法':<10}{'Spearman':>10}{'p':>8}{'置换p':>8}{'Bootstrap 95%CI':>22}")
    for label, ser in criteria:
        d = base_idx.join(ser.rename("h")).join(lb["labse_similarity"], how="left")
        for nm, col in [("MiniLM", "similarity_score"), ("LaBSE", "labse_similarity")]:
            x = d[col].values.astype(float); y = d["h"].values.astype(float)
            ok = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[ok], y[ok]
            sp = stats.spearmanr(x, y); ci = boot_ci(x, y)
            print(f"{label:<18}{len(x):>4}{nm:<10}{sp.statistic:>10.4f}{sp.pvalue:>8.4f}"
                  f"{perm_p(x, y):>8.4f}{'[%.3f, %.3f]' % (ci[0], ci[1]):>22}")
        r = stats.spearmanr(d["len_en_tok"], d["h"])
        print(f"{'':<18}{'':>4}{'英文长度vs基准':<10}{r.statistic:>10.4f}{r.pvalue:>8.4f}")

    print("\n" + "=" * 78)
    print("建议：表3 主口径用“全部3人均值”，并把“仅独立2人均值”作为敏感性分析。")


if __name__ == "__main__":
    main()
