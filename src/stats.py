# analysis.py
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# ============ 1. 重测信度 ============

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

print("=" * 60)
print("1. 人机一致性检验（AI vs 人工）")
print("=" * 60)

df1 = pd.read_csv(os.path.join(DATA, 'misreading_scored_round1.csv'))
df2 = pd.read_csv(os.path.join(DATA, 'misreading_scored_round2.csv'))

only1 = set(df1['pair_id']) - set(df2['pair_id'])
only2 = set(df2['pair_id']) - set(df1['pair_id'])
if only1 or only2:
    print(f"⚠️ 仅在round1: {sorted(only1)}")
    print(f"⚠️ 仅在round2: {sorted(only2)}\n")

common_ids = sorted(set(df1['pair_id']) & set(df2['pair_id']))
df1_c = df1[df1['pair_id'].isin(common_ids)].copy()
df2_c = df2[df2['pair_id'].isin(common_ids)].copy()

merged = pd.merge(df1_c[['pair_id', 'human_score']],
                  df2_c[['pair_id', 'human_score']],
                  on='pair_id', suffixes=('_r1', '_r2'),
                  validate='one_to_one').dropna()

r, p = pearsonr(merged['human_score_r1'], merged['human_score_r2'])
rho, p_s = spearmanr(merged['human_score_r1'], merged['human_score_r2'])
print(f"Pearson r = {r:.4f}, p = {p:.4f}")
print(f"Spearman rho = {rho:.4f}, p = {p_s:.4f}")
print(f"共同样本数 = {len(merged)}")

try:
    import pingouin as pg
    paired = pd.merge(
        df1_c[['pair_id', 'human_score']],
        df2_c[['pair_id', 'human_score']],
        on='pair_id', suffixes=('_r1', '_r2'), validate='one_to_one'
    ).dropna()
    icc_data = pd.concat([
        paired[['pair_id', 'human_score_r1']].rename(columns={'human_score_r1': 'human_score'}).assign(rater='r1'),
        paired[['pair_id', 'human_score_r2']].rename(columns={'human_score_r2': 'human_score'}).assign(rater='r2'),
    ])
    icc = pg.intraclass_corr(
        data=icc_data,
        targets='pair_id',
        raters='rater',
        ratings='human_score'
    )
    print("\nICC 结果:")
    print(icc.to_string())  # 改为 to_string()，不指定列名
except Exception as e:
    print(f"\nICC 计算失败，跳过: {e}")


# ============ 2. 合并所有方法 ============
print("\n" + "=" * 60)
print("2. 合并所有方法结果")
print("=" * 60)

df_main = pd.read_csv(os.path.join(DATA, 'misreading_scored_round2.csv'))
# 主口径：round2/3/4 三轮盲评均值（与论文表 4 一致）
_rs = []
for _k in (2, 3, 4):
    _d = pd.read_csv(os.path.join(DATA, 'misreading_scored_round%d.csv' % _k))
    _rs.append(_d.set_index('pair_id')['human_score'])
_crit = pd.concat(_rs, axis=1).mean(axis=1).rename('human_score')
df_main = df_main.drop(columns=['human_score']).merge(
    _crit.reset_index(), on='pair_id', how='left')
df_labse = pd.read_csv(os.path.join(DATA, 'labse_results.csv'))

df_all = df_main.merge(df_labse[['pair_id', 'labse_similarity']],
                       on='pair_id', how='left', validate='one_to_one')

if os.path.exists(os.path.join(OUT, 'mt_results.csv')):
    df_mt = pd.read_csv(os.path.join(OUT, 'mt_results.csv'))
    df_all = df_all.merge(df_mt[['pair_id', 'mt_similarity']],
                          on='pair_id', how='left', validate='one_to_one')
    print("已合并 MT 基线")

print(f"原始行数: {len(df_main)}，合并后行数: {len(df_all)}")
assert len(df_all) == len(df_main), "行数异常"

df_all.to_csv(os.path.join(DATA, 'all_results_merged.csv'), index=False, encoding='utf-8-sig')


# ============ 3. 相关系数 + 置换检验 ============
print("\n" + "=" * 60)
print("3. 相关系数与置换检验")
print("=" * 60)


def corr_with_perm(df, col_x, col_y, n_perm=10000, seed=42):
    data = df[[col_x, col_y]].dropna()
    n = len(data)
    print(f"  有效样本数: {n}")
    if n < 10:
        print(f"  ⚠️ 样本数 < 10，相关系数不可信")

    x, y = data[col_x].values, data[col_y].values
    spearman_r, spearman_p = spearmanr(x, y)
    pearson_r, pearson_p = pearsonr(x, y)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        perm_r, _ = spearmanr(x, perm_y)
        if abs(perm_r) >= abs(spearman_r):
            count += 1
    perm_p = (count + 1) / (n_perm + 1)

    return {
        'n': n, 'spearman_r': round(spearman_r, 4),
        'spearman_p': round(spearman_p, 4), 'pearson_r': round(pearson_r, 4),
        'pearson_p': round(pearson_p, 4), 'perm_p': round(perm_p, 4),
    }


results = {}
print("\n本文方法:")
results['ours'] = corr_with_perm(df_all, 'similarity_score', 'human_score')
print(results['ours'])

print("\nLaBSE:")
results['labse'] = corr_with_perm(df_all, 'labse_similarity', 'human_score')
print(results['labse'])

if 'mt_similarity' in df_all.columns:
    print("\nMT+SBERT:")
    results['mt'] = corr_with_perm(df_all, 'mt_similarity', 'human_score')
    print(results['mt'])

print("\n" + "-" * 60)
print("Bonferroni 校正后 p 值：")
n_methods = len(results)
for name, r in results.items():
    print(f"{name}: perm_p = {r['perm_p']:.4f}, 校正后 = {min(r['perm_p'] * n_methods, 1.0):.4f}")

# 随机基线
print("\n" + "-" * 60)
print("随机基线（1000次）：")
y_true = df_all['human_score'].dropna().values
n_rand = len(y_true)
random_rs = []
for i in range(1000):
    rs = np.random.default_rng(1000 + i).uniform(0, 1, n_rand)
    r_i, _ = spearmanr(rs, y_true)
    random_rs.append(r_i)
random_r_mean = np.mean(random_rs)
random_r_ci = (np.percentile(random_rs, 2.5), np.percentile(random_rs, 97.5))
print(f"随机基线: Spearman r 均值 = {random_r_mean:.4f}, "
      f"95% 区间 = [{random_r_ci[0]:.4f}, {random_r_ci[1]:.4f}]")


# ============ 4. 偏相关分析（控制英文长度）============
if 'len_en_tok' in df_all.columns:
    print("\n" + "=" * 60)
    print("4. 偏相关分析（控制英文长度）")
    print("=" * 60)
    try:
        import pingouin as pg
        partial_df = df_all.dropna(subset=['similarity_score', 'human_score', 'len_en_tok'])
        partial = pg.partial_corr(
            data=partial_df,
            x='similarity_score', y='human_score',
            covar='len_en_tok', method='spearman'
        )
        print(partial.to_string())
    except Exception as e:
        print(f"偏相关计算失败: {e}")


# ============ 5. 论文表2数据 ============
print("\n" + "=" * 60)
print("5. 论文表2数据")
print("=" * 60)

n_valid = df_all[['similarity_score', 'human_score']].dropna().shape[0]

rows = [{
    '方法': '随机基线', 'Spearman r': f"{random_r_mean:.4f}",
    'Pearson r': '—', '置换检验p': '—', '样本数': n_valid,
}]
for name, label in [('ours', '本文方法'), ('labse', 'LaBSE'), ('mt', 'MT+SBERT')]:
    if name in results:
        r = results[name]
        rows.append({
            '方法': label, 'Spearman r': r['spearman_r'],
            'Pearson r': r['pearson_r'], '置换检验p': r['perm_p'],
            '样本数': r['n'],
        })

df_table2 = pd.DataFrame(rows)
print(df_table2.to_string(index=False))
df_table2.to_csv(os.path.join(DATA, 'table2_results.csv'), index=False, encoding='utf-8-sig')
from scipy.stats import norm

def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)
    lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
    return lo, hi

print("\n" + "=" * 60)
print("Fisher 95% 置信区间")
print("=" * 60)
for name, label in [('ours', '本文方法'), ('labse', 'LaBSE')]:
    if name in results:
        r = results[name]
        lo, hi = fisher_ci(r['spearman_r'], r['n'])
        print(f"{label}: r = {r['spearman_r']:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}]")

# 分层分析
print("\n" + "=" * 60)
print("分层分析：短文本子集（< 200 token）")
print("=" * 60)
df_short = df_all[df_all['len_en_tok'] < 200]
print(f"短文本子集样本数: {len(df_short)}")
results_short_ours = corr_with_perm(df_short, 'similarity_score', 'human_score')
results_short_labse = corr_with_perm(df_short, 'labse_similarity', 'human_score')
print("短文本-本文方法:", results_short_ours)
print("短文本-LaBSE:", results_short_labse)
# ============ 6. LaBSE 偏相关分析 ============
print("\n" + "=" * 60)
print("6. LaBSE 偏相关分析（控制英文长度）")
print("=" * 60)

try:
    import pingouin as pg
    partial_df_labse = df_all.dropna(subset=['labse_similarity', 'human_score', 'len_en_tok'])
    if len(partial_df_labse) >= 10:
        partial_labse = pg.partial_corr(
            data=partial_df_labse,
            x='labse_similarity', y='human_score',
            covar='len_en_tok', method='spearman'
        )
        print("LaBSE偏相关结果：")
        print(partial_labse.to_string())
    else:
        print(f"有效样本 {len(partial_df_labse)}，不足10，跳过")
except ImportError:
    print("pingouin未安装，跳过")
except Exception as e:
    print(f"偏相关计算失败: {e}")


# ============ 7. Bootstrap置信区间 ============
print("\n" + "=" * 60)
print("7. Bootstrap置信区间（10000次）")
print("=" * 60)

def bootstrap_ci(x, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rs.append(r)
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)

for col, label in [('similarity_score', 'MiniLM'), ('labse_similarity', 'LaBSE')]:
    data = df_all[[col, 'human_score']].dropna()
    r, _ = spearmanr(data[col], data['human_score'])
    lo, hi = bootstrap_ci(data[col].values, data['human_score'].values)
    print(f"{label}: r = {r:.4f}, Bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")


# ============ 8. 与Fisher z变换对比 ============
print("\n" + "=" * 60)
print("8. 与Fisher z变换对比")
print("=" * 60)

from scipy.stats import norm

def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)
    lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
    return lo, hi

for col, label in [('similarity_score', 'MiniLM'), ('labse_similarity', 'LaBSE')]:
    data = df_all[[col, 'human_score']].dropna()
    r, _ = spearmanr(data[col], data['human_score'])
    lo_f, hi_f = fisher_ci(r, len(data))
    print(f"{label}: Fisher 95% CI = [{lo_f:.4f}, {hi_f:.4f}]")

    from scipy.stats import spearmanr

    rho, p = spearmanr(df_all['len_cn_tok'], df_all['labse_similarity'])
    print(f"LaBSE中文长度: rho={rho:.4f}, p={p:.4f}")