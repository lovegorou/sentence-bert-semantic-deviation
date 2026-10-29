import os
# 如需国内镜像，运行前设置环境变量 HF_ENDPOINT

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# misreading.py
import re
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util

_LOCAL = os.path.join(MODELS, 'MiniLM')
MODEL_NAME = _LOCAL if os.path.isdir(_LOCAL) else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EXCEL_PATH = os.path.join(DATA, 'annotation_template.xlsx')
OUTPUT_CSV = os.path.join(DATA, 'misreading_results.csv')

POETS = ['李白', '王维', '韩愈', '枚乘', '杜甫', '白居易', '杜牧', '李商隐', '苏轼', '陶渊明']
FALLBACK_USED = []


def clean_chinese(s):
    s = str(s)
    s = re.sub(r'《[^》]+》', '', s)
    s = s.strip().strip('。，、；;：: ')
    for p in POETS:
        if s.endswith(p):
            s = s[:-len(p)].strip('。，、；;：: ')
    return s.strip()


def encode_long_text(text, model, tokenizer, text_id=None,
                     max_len=128, stride=64, min_chunk=10):
    """长文本分段编码取平均。token id 层切分，避免 decode→encode 中文空格噪声。"""
    token_ids = tokenizer.encode(text, truncation=False, add_special_tokens=False)

    if len(token_ids) <= max_len:
        return model.encode(text, convert_to_tensor=True)

    chunk_ids_list = []
    dropped = 0
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start:start + max_len]
        if len(chunk_ids) < min_chunk:
            dropped = len(chunk_ids)
            break
        chunk_ids_list.append(chunk_ids)

    if dropped > 0:
        print(f"  [pair {text_id}] 尾部 {dropped} token 被丢弃（< {min_chunk}）")

    max_chunk_len = max(len(c) for c in chunk_ids_list)
    input_ids = torch.zeros(len(chunk_ids_list), max_chunk_len, dtype=torch.long)
    attention_mask = torch.zeros(len(chunk_ids_list), max_chunk_len, dtype=torch.long)
    for i, c in enumerate(chunk_ids_list):
        input_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
        attention_mask[i, :len(c)] = 1

    device = next(model.parameters()).device
    features = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
    }

    try:
        with torch.no_grad():
            out = model(features)              # 走完整 pipeline
        emb = out['sentence_embedding']
        return emb.mean(dim=0)
    except Exception as e:
        print(f"  [pair {text_id}] 底层API失败，回退 decode: {e}")
        FALLBACK_USED.append(text_id)
        chunk_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunk_ids_list]
        embs = model.encode(chunk_texts, convert_to_tensor=True)
        return embs.mean(dim=0)


# ============ 1. 加载模型 ============
print("加载模型...")
model = SentenceTransformer(MODEL_NAME, local_files_only=os.path.isdir(_LOCAL))
model.max_seq_length = 128
tokenizer = model.tokenizer

# ============ 2. 读取数据 ============
print("读取 Excel...")
df = pd.read_excel(EXCEL_PATH)

print("\nwork_title 唯一值：", df['work_title'].unique())
print("paragraph_id 分布：", sorted(df['paragraph_id'].unique()))

# 主要过滤：text 和 source_text 都非空
# 额外防御：排除 Seafarer（按 paragraph_id 或 text 内容）
df_valid = df[
    df['text'].notna() &
    (df['text'].astype(str).str.strip() != '') &
    df['source_text'].notna() &
    (df['source_text'].astype(str).str.strip() != '')
].copy()

print(f"\n有效段落数: {len(df_valid)}")

# ============ 3. 构建文本对（方案A：整段 vs 整段）============
pairs = []
pair_id = 0
for idx, row in df_valid.iterrows():
    english = str(row['text']).strip()
    raw_cn = str(row['source_text']).replace('；', ';')
    chinese_sentences = []
    for s in raw_cn.split(';'):
        c = clean_chinese(s)
        if c:
            chinese_sentences.append(c)
    chinese_combined = ' '.join(chinese_sentences)

    pair_id += 1
    pairs.append({
        'pair_id': pair_id,
        'paragraph_id': row['paragraph_id'],
        'work_title': row['work_title'],
        'text_en': english,
        'text_cn': chinese_combined,
    })

df_pairs = pd.DataFrame(pairs)
print(f"文本对数: {len(df_pairs)}")

# ============ 4. Token 长度检查 ============
len_en = [len(tokenizer.encode(t, truncation=False)) for t in df_pairs['text_en']]
len_cn = [len(tokenizer.encode(t, truncation=False)) for t in df_pairs['text_cn']]
print(f"\n英文 token — 最长: {max(len_en)}, 95分位: {int(np.percentile(len_en, 95))}, "
      f"超过128的: {sum(l > 128 for l in len_en)}")
print(f"中文 token — 最长: {max(len_cn)}, 95分位: {int(np.percentile(len_cn, 95))}, "
      f"超过128的: {sum(l > 128 for l in len_cn)}")

# ============ 5. 分段编码 ============
print("\n分段编码英文...")
emb_en_list = []
for i, t in enumerate(df_pairs['text_en']):
    emb_en_list.append(encode_long_text(t, model, tokenizer, text_id=f"en_{i+1}"))
    if (i + 1) % 5 == 0:
        print(f"  已处理 {i+1}/{len(df_pairs)}")
emb_en = torch.stack(emb_en_list)

print("分段编码中文...")
emb_cn_list = []
for i, t in enumerate(df_pairs['text_cn']):
    emb_cn_list.append(encode_long_text(t, model, tokenizer, text_id=f"cn_{i+1}"))
emb_cn = torch.stack(emb_cn_list)

if FALLBACK_USED:
    print(f"\n⚠️ 回退列表：{FALLBACK_USED}")
else:
    print("\n✅ 全部走底层 token 层编码\n")

# ============ 6. 相似度 ============
sims = util.cos_sim(emb_en, emb_cn).diag().cpu().numpy()
sims = np.clip(sims, 0.0, 1.0)

df_pairs['similarity_score'] = np.round(sims, 4)
df_pairs['deviation_score'] = np.round(1 - sims, 4)
df_pairs['len_en_tok'] = len_en
df_pairs['len_cn_tok'] = len_cn

# ============ 7. 长度敏感性检验 ============
print("=" * 60)
print("长度敏感性检验")
print("=" * 60)
rho_en, p_en = spearmanr(df_pairs['len_en_tok'], df_pairs['similarity_score'])
rho_cn, p_cn = spearmanr(df_pairs['len_cn_tok'], df_pairs['similarity_score'])
print(f"英文 token 长度 vs 相似度: rho = {rho_en:.4f}, p = {p_en:.4f}")
print(f"中文 token 长度 vs 相似度: rho = {rho_cn:.4f}, p = {p_cn:.4f}")
if p_en < 0.05:
    print("⚠️ 英文长度与相似度显著相关，analysis.py 将做偏相关分析")

# ============ 8. 输出 ============
df_pairs.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n完成！输出: {OUTPUT_CSV}")
print(df_pairs[['pair_id', 'paragraph_id', 'work_title',
                'similarity_score', 'deviation_score']].to_string())