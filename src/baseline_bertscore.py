# -*- coding: utf-8 -*-
"""baseline_bertscore.py —— BERTScore 基线

以英文文本为候选、中文原典为参照，用多语言 BERT 计算 BERTScore。
与句向量方案属不同范式（词级最优对齐 vs 整句向量余弦），故作为独立基线。

依赖（需另行安装，未列入主 requirements）：
    pip install --no-deps bert-score
    # bert-score 0.3.10/0.3.11 的元数据损坏，新版 pip 会拒绝；0.3.12 已修复，
    # 若仍报错可加 --no-deps 跳过依赖解析（torch/transformers/numpy 本项目已具备）

用法：
    python src/baseline_bertscore.py
输出：
    data/bertscore_results.csv
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import DATA, OUT  # noqa: E402

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')  # 国内镜像

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODEL = 'bert-base-multilingual-cased'


def main():
    d = pd.read_csv(os.path.join(DATA, 'misreading_results.csv'))
    crit = []
    for k in (2, 3, 4):
        s = pd.read_csv(os.path.join(DATA, 'misreading_scored_round%d.csv' % k))
        crit.append(s.set_index('pair_id')['human_score'])
    d['human'] = d['pair_id'].map(pd.concat(crit, axis=1).mean(axis=1))
    d = d.dropna(subset=['human']).copy()
    print('样本数:', len(d))

    from bert_score import score as bs_score
    print('计算 BERTScore（%s）...' % MODEL)
    P, R, F1 = bs_score(
        d['text_en'].astype(str).tolist(),
        d['text_cn'].astype(str).tolist(),
        model_type=MODEL, lang='en', verbose=False, idf=False,
        batch_size=4, device='cpu', rescale_with_baseline=False)
    d['bertscore_p'] = P.numpy().round(4)
    d['bertscore_r'] = R.numpy().round(4)
    d['bertscore_f1'] = F1.numpy().round(4)

    rng = np.random.default_rng(42)

    def perm_p(x, y, n=10000):
        r0 = spearmanr(x, y).statistic
        return sum(1 for _ in range(n)
                   if abs(spearmanr(x, rng.permutation(y)).statistic) >= abs(r0)) / n

    def boot_ci(x, y, n=10000):
        idx = np.arange(len(x)); out = []
        for _ in range(n):
            s = rng.choice(idx, len(idx), True)
            if len(set(y[s])) < 3:
                continue
            out.append(spearmanr(x[s], y[s]).statistic)
        return np.percentile(out, [2.5, 97.5])

    print()
    for col in ['bertscore_f1', 'bertscore_p', 'bertscore_r']:
        rho, _ = spearmanr(d[col], d['human'])
        pp = perm_p(d[col].values, d['human'].values, 2000)
        lo, hi = boot_ci(d[col].values, d['human'].values, 2000)
        print('%-14s rho=%+.4f  perm_p=%.4f  Bootstrap[%+.3f, %+.3f]'
              % (col, rho, pp, lo, hi))

    out = os.path.join(DATA, 'bertscore_results.csv')
    d[['pair_id', 'bertscore_p', 'bertscore_r', 'bertscore_f1', 'human']].to_csv(
        out, index=False, encoding='utf-8-sig')
    print('\n已保存:', out)


if __name__ == '__main__':
    main()
