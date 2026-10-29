import os

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = os.path.join(MODELS, 'MiniLM')
INPUT_CSV = os.path.join(DATA, 'misreading_scored_round1.csv')
MAX_CHARS = 4500

translator = GoogleTranslator(source='en', target='zh-CN')
model = SentenceTransformer(MODEL_NAME, local_files_only=True)
model.max_seq_length = 128

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['text_en', 'text_cn']).copy()
df['text_en'] = df['text_en'].astype(str)
df['text_cn'] = df['text_cn'].astype(str)

translated_texts = []
failed = []
too_long = []

print("开始翻译...")
for idx, row in df.iterrows():
    text = row['text_en']
    if len(text) > MAX_CHARS:
        print(f"pair_id {row['pair_id']} 文本过长 ({len(text)} 字符)，跳过")
        translated_texts.append('[EMPTY]')
        failed.append(row['pair_id'])
        too_long.append(row['pair_id'])
        continue

    success = False
    for attempt in range(2):
        try:
            translated = translator.translate(text)
            if translated and translated.strip():
                translated_texts.append(translated)
                success = True
                break
        except Exception as e:
            print(f"pair_id {row['pair_id']} 第{attempt+1}次失败: {e}")
            time.sleep(1)
    if not success:
        translated_texts.append('[EMPTY]')
        failed.append(row['pair_id'])
    time.sleep(0.5)

print("批量编码翻译结果...")
emb_trans = model.encode(translated_texts, convert_to_tensor=True,
                        batch_size=64, show_progress_bar=True)
print("批量编码原始中文...")
emb_cn = model.encode(df['text_cn'].tolist(), convert_to_tensor=True,
                     batch_size=64, show_progress_bar=True)

sims = util.cos_sim(emb_trans, emb_cn).diag().cpu().numpy()
sims = np.clip(sims, 0.0, 1.0)

df_mt = pd.DataFrame({
    'pair_id': df['pair_id'].values,
    'mt_similarity': np.round(sims, 4),
    'mt_deviation': np.round(1 - sims, 4),
})

df_mt.loc[df_mt['pair_id'].isin(failed), ['mt_similarity', 'mt_deviation']] = np.nan
df_mt.to_csv(os.path.join(OUT, 'mt_results.csv'), index=False, encoding='utf-8-sig')

if failed:
    pd.DataFrame({'pair_id': failed}).to_csv(os.path.join(OUT, 'mt_failed.csv'), index=False, encoding='utf-8-sig')
    print(f"失败 {len(failed)} 对（超长 {len(too_long)}）")

print(f"\nMT+SBERT 完成！成功 {len(df_mt) - len(failed)} 对，失败 {len(failed)} 对")