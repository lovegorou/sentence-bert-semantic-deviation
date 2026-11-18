# -*- coding: utf-8 -*-
"""Chunked BERTScore sensitivity analysis.

The original baseline feeds full texts to mBERT, which truncates at 512 tokens.
This script splits the English candidate into non-overlapping 400-token chunks,
scores each chunk against the full Chinese source, and aggregates
precision/recall/F1 weighted by chunk length. It writes the per-pair scores and
reports the correlations used in the paper's Table 4/5.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
from scipy import stats
from transformers import AutoTokenizer
from bert_score import score as bs_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'all_results_merged.csv')
OUT = os.path.join(ROOT, 'outputs', 'bertscore_chunked_results.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(DATA)
tok = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
MAX, MIN = 400, 10

rows = []
for _, row in df.iterrows():
    en, cn = str(row['text_en']), str(row['text_cn'])
    ids = tok(en, add_special_tokens=False)['input_ids']
    chunks = []
    for start in range(0, len(ids), MAX):
        piece = ids[start:start + MAX]
        if len(piece) >= MIN:
            chunks.append(tok.decode(piece, skip_special_tokens=True))
    Ps, Rs, Fs, Ws = [], [], [], []
    for ch in chunks:
        P, R, F = bs_score([ch], [cn], model_type='bert-base-multilingual-cased',
                           lang='en', idf=False, batch_size=1, device='cpu',
                           rescale_with_baseline=False, verbose=False)
        Ps.append(float(P[0])); Rs.append(float(R[0])); Fs.append(float(F[0]))
        Ws.append(len(tok(ch, add_special_tokens=False)['input_ids']))
    Ws = np.array(Ws, dtype=float); Ws = Ws / Ws.sum()
    rows.append(dict(pair_id=int(row['pair_id']), n_chunks=len(chunks),
                     chunked_p=float(np.dot(Ps, Ws)),
                     chunked_r=float(np.dot(Rs, Ws)),
                     chunked_f1=float(np.dot(Fs, Ws))))
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out = out.merge(df[['pair_id', 'human_score', 'len_en_tok']], on='pair_id')
out.to_csv(OUT, index=False, encoding='utf-8-sig')


def rho(x, y):
    return stats.spearmanr(x, y)


def partial(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    t = r * np.sqrt((len(x) - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), len(x) - 3)


for col in ['chunked_p', 'chunked_r', 'chunked_f1']:
    print('%-10s vs human rho=%+.4f p=%.4f | vs length rho=%+.4f p=%.4f | partial=%+.4f p=%.4f'
          % (col, rho(out[col], out.human_score).statistic, rho(out[col], out.human_score).pvalue,
             rho(out[col], out.len_en_tok).statistic, rho(out[col], out.len_en_tok).pvalue,
             *partial(out[col].values, out.human_score.values, out.len_en_tok.values)))
print('saved', OUT)
