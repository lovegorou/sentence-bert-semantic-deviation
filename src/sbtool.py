# -*- coding: utf-8 -*-
"""Unified command-line entry for the semantic-deviation tool.

Usage
-----
    python src/sbtool.py --encoder minilm --input data.csv --output scores.csv
    python src/sbtool.py --encoder labse  --input data.csv --output scores.csv

Input CSV must contain: pair_id, text_en, text_cn.
Output CSV contains: pair_id, similarity, deviation, len_en_tok, len_cn_tok.

MiniLM uses token-id-level sliding windows (mean pooling); LaBSE uses decoded
text chunks so that each chunk keeps its [CLS]/[SEP] markers (CLS pooling).
"""
import argparse
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
import pandas as pd

MODEL_INFO = {
    'minilm': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 128),
    'labse': ('sentence-transformers/LaBSE', 512),
}


def local_model_dir(encoder, explicit=None):
    if explicit:
        return explicit
    try:
        from paths import MODELS
    except Exception:
        MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    names = {'minilm': ['MiniLM', 'minilm', 'paraphrase-multilingual-MiniLM-L12-v2'],
             'labse': ['LaBSE', 'labse']}[encoder]
    for n in names:
        p = os.path.join(MODELS, n)
        if os.path.isdir(p):
            return p
    return None


def encode_chunked(model, tokenizer, text, max_len, stride=64, min_chunk=10, text_chunks=False):
    import torch
    ids = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    if len(ids) <= max_len:
        return model.encode([text], convert_to_numpy=True, batch_size=1)[0]
    chunks = []
    for start in range(0, len(ids), stride):
        piece = ids[start:start + max_len]
        if len(piece) < min_chunk:
            break
        chunks.append(piece)
    if not chunks:
        return model.encode([text], convert_to_numpy=True, batch_size=1)[0]
    if text_chunks:
        texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
        embs = model.encode(texts, convert_to_numpy=True, batch_size=8)
        return embs.mean(axis=0)
    max_chunk = max(len(c) for c in chunks)
    input_ids = torch.zeros(len(chunks), max_chunk, dtype=torch.long)
    attention = torch.zeros(len(chunks), max_chunk, dtype=torch.long)
    for i, c in enumerate(chunks):
        input_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
        attention[i, :len(c)] = 1
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model({'input_ids': input_ids.to(device),
                     'attention_mask': attention.to(device)})
    return out['sentence_embedding'].mean(dim=0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description='Multilingual semantic deviation tool')
    ap.add_argument('--encoder', choices=['minilm', 'labse'], default='minilm')
    ap.add_argument('--input', required=True, help='CSV with pair_id, text_en, text_cn')
    ap.add_argument('--output', required=True, help='output CSV path')
    ap.add_argument('--model-dir', default=None, help='local model directory (optional)')
    ap.add_argument('--stride', type=int, default=64)
    ap.add_argument('--min-chunk', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='only process the first N rows (0 = all)')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    need = {'pair_id', 'text_en', 'text_cn'}
    if not need.issubset(df.columns):
        sys.exit('input CSV must contain: ' + ', '.join(sorted(need)))
    if args.limit:
        df = df.head(args.limit).copy()

    hf_name, max_len = MODEL_INFO[args.encoder]
    local = local_model_dir(args.encoder, args.model_dir)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(local or hf_name, local_files_only=bool(local))
    model.max_seq_length = max_len
    tok = model.tokenizer
    text_chunks = (args.encoder == 'labse')

    rows = []
    for _, r in df.iterrows():
        ve = encode_chunked(model, tok, str(r['text_en']), max_len, args.stride,
                            args.min_chunk, text_chunks)
        vc = encode_chunked(model, tok, str(r['text_cn']), max_len, args.stride,
                            args.min_chunk, text_chunks)
        ve = ve / np.linalg.norm(ve)
        vc = vc / np.linalg.norm(vc)
        sim = float(np.clip(np.dot(ve, vc), 0.0, 1.0))
        rows.append(dict(pair_id=r['pair_id'], similarity=round(sim, 4),
                         deviation=round(1 - sim, 4),
                         len_en_tok=len(tok.encode(str(r['text_en']), truncation=False)),
                         len_cn_tok=len(tok.encode(str(r['text_cn']), truncation=False))))
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding='utf-8-sig')
    print('saved', args.output, 'rows', len(out))


if __name__ == '__main__':
    main()
