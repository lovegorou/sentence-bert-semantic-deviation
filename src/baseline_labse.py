# baseline_labse.py —— LaBSE 基线
import os

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

_LOCAL = os.path.join(MODELS, 'LaBSE')
MODEL_NAME = _LOCAL if os.path.isdir(_LOCAL) else 'sentence-transformers/LaBSE'
INPUT_CSV = os.path.join(DATA, 'misreading_results.csv')
OUTPUT_CSV = os.path.join(DATA, 'labse_results.csv')

print("加载 LaBSE 模型（本地路径）...")
model = SentenceTransformer(MODEL_NAME, local_files_only=os.path.isdir(_LOCAL))
model.max_seq_length = 512

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['text_en', 'text_cn']).copy()
df['text_en'] = df['text_en'].astype(str)
df['text_cn'] = df['text_cn'].astype(str)

tokenizer = model.tokenizer
lengths = [len(tokenizer.encode(t, truncation=False)) for t in df['text_en']]
print(f"英文最长 token: {max(lengths)}, 超过512的: {sum(l > 512 for l in lengths)}")

print("批量编码英文...")
emb_en = model.encode(df['text_en'].tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)
print("批量编码中文...")
emb_cn = model.encode(df['text_cn'].tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)

sims = util.cos_sim(emb_en, emb_cn).diag().cpu().numpy()
sims = np.clip(sims, 0.0, 1.0)

df_labse = pd.DataFrame({
    'pair_id': df['pair_id'].values,
    'labse_similarity': np.round(sims, 4),
    'labse_deviation': np.round(1 - sims, 4),
})
df_labse.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\nLaBSE 完成！共 {len(df_labse)} 对")
print(df_labse.to_string())