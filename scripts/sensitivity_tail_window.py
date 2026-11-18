# -*- coding: utf-8 -*-
"""Sensitivity check: MiniLM chunking with a loop that stops at the text end.

The original pipeline uses `for start in range(0, len(ids), stride)`, which
keeps emitting overlapping tail windows until the remaining piece is shorter
than 10 tokens. This script uses `while start < len(ids)` and breaks once the
window reaches the end, removing the duplicated tail. It reads the old
(fixed-stride) scores from `data/minilm_old_results.csv`, recomputes the
corrected scores, and writes the per-pair comparison to
`data/minilm_fixed_results.csv`.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'all_results_merged.csv')
OLD = os.path.join(ROOT, 'data', 'minilm_old_results.csv')
OUT = os.path.join(ROOT, 'data', 'minilm_fixed_results.csv')

from sentence_transformers import SentenceTransformer
MODEL = os.path.join(ROOT, 'models', 'MiniLM')
model = SentenceTransformer(MODEL if os.path.isdir(MODEL) else
                            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                            local_files_only=os.path.isdir(MODEL))
model.max_seq_length = 128
tok = model.tokenizer


def embed(text):
    ids = tok.encode(str(text), truncation=False, add_special_tokens=False)
    if len(ids) <= 128:
        return model.encode([str(text)], convert_to_numpy=True)[0]
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + 128, len(ids))
        piece = ids[start:end]
        if len(piece) >= 10:
            chunks.append(piece)
        if end >= len(ids):
            break
        start += 64
    mx = max(len(c) for c in chunks)
    ids_t = torch.zeros(len(chunks), mx, dtype=torch.long)
    att = torch.zeros(len(chunks), mx, dtype=torch.long)
    for i, c in enumerate(chunks):
        ids_t[i, :len(c)] = torch.tensor(c)
        att[i, :len(c)] = 1
    with torch.no_grad():
        out = model({'input_ids': ids_t, 'attention_mask': att})
    return out['sentence_embedding'].mean(dim=0).cpu().numpy()


df = pd.read_csv(DATA)
old = pd.read_csv(OLD)
df = df.drop(columns=['similarity_score']).merge(old, on='pair_id')

rows = []
for _, r in df.iterrows():
    ve, vc = embed(r['text_en']), embed(r['text_cn'])
    ve, vc = ve / np.linalg.norm(ve), vc / np.linalg.norm(vc)
    sim = float(np.clip(np.dot(ve, vc), 0, 1))
    rows.append(dict(pair_id=r['pair_id'], old_sim=r['old_sim'],
                     fixed_sim=round(sim, 4), delta=round(sim - r['old_sim'], 4)))
out = pd.DataFrame(rows)
out['old_rank'] = out['old_sim'].rank(ascending=False, method='min').astype(int)
out['new_rank'] = out['fixed_sim'].rank(ascending=False, method='min').astype(int)
out = out.merge(df[['pair_id', 'human_score']], on='pair_id')
out.to_csv(OUT, index=False, encoding='utf-8-sig')
print(out.to_string(index=False))

d = out.merge(df[['pair_id', 'len_en_tok']], on='pair_id')
h = d['human_score'].values
for col in ['old_sim', 'fixed_sim']:
    print(col, 'rho=%.4f p=%.4f | len rho=%.4f p=%.4f' % (
        stats.spearmanr(d[col], h).statistic, stats.spearmanr(d[col], h).pvalue,
        stats.spearmanr(d[col], d['len_en_tok']).statistic,
        stats.spearmanr(d[col], d['len_en_tok']).pvalue))
print('saved', OUT)
