# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

"""
perf_benchmark.py —— 工具性能基准测试
========================================
测量内容：
  1. 模型加载耗时
  2. 单对文本编码延迟（短文本 / 中等 / 超长，分别测，体现分段开销）
  3. 批处理吞吐量（14 对整体处理，pairs/s 与 token/s）
  4. 峰值内存占用（RSS）
  5. 两个编码器（MiniLM / LaBSE）对照

用法：
    python perf_benchmark.py
输出：
    控制台表格 + perf_results.csv（UTF-8，可直接贴进论文 §3.5）
注意：
    - 需在有模型权重的目录下运行；LaBSE 默认读本地路径，若不存在会尝试 HF 名称
    - 首次运行会下载 MiniLM（约 117MB）
    - 结果受机器负载影响，建议关闭其他程序后连跑 3 次取中位数
"""
import os
import sys
import time
import threading
import platform
import statistics as st

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import psutil

# ============ 配置区 ============
# 本地权重路径（推荐）：有本地模型时填这里，避免联网
# 也可用环境变量 MODEL_MINILM_LOCAL / MODEL_LABSE_LOCAL 覆盖
MODEL_MINILM_LOCAL = os.environ.get("MODEL_MINILM_LOCAL", "")
MODEL_LABSE_LOCAL = os.environ.get(
    "MODEL_LABSE_LOCAL", os.path.join(MODELS, "LaBSE"))

# 在线模型名（本地路径为空或不存在时回退）
MODEL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_LABSE_HF = "sentence-transformers/LaBSE"

MAX_LEN_MINILM = 128
MAX_LEN_LABSE = 512
STRIDE = 64
MIN_CHUNK = 10

WARMUP = 3          # 预热次数
REPEAT = 10         # 每个样本重复测量次数，取中位数
CORPUS_CSV = os.path.join(DATA, "misreading_results.csv")
OUT_CSV = os.path.join(OUT, "perf_results.csv")
# ================================


class RSSTracker:
    """后台采样进程 RSS，记录峰值。"""

    def __init__(self, interval=0.05):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.peak = self.proc.memory_info().rss
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=1)

    def now_mb(self):
        return self.proc.memory_info().rss / 1024 / 1024

    def peak_mb(self):
        return self.peak / 1024 / 1024


def pick_device_note():
    cpu = platform.processor() or "unknown"
    total = psutil.virtual_memory().total / 1024 ** 3
    return cpu, total, platform.python_version()


def encode_long_text(text, model, tokenizer, max_len=MAX_LEN_MINILM,
                     stride=STRIDE, min_chunk=MIN_CHUNK):
    """与 misreading.py 一致的分段编码：重叠滑窗 + 池化向量取均值。"""
    import torch
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return model.encode(text, convert_to_numpy=True)
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + max_len, len(ids))
        piece = ids[start:end]
        if len(piece) >= min_chunk:
            chunks.append(piece)
        if end >= len(ids):
            break
        start += stride
    if not chunks:
        return model.encode(text, convert_to_numpy=True)
    # 整批零填充后一次前向，与 misreading.py 的做法一致
    enc = tokenizer.pad({"input_ids": chunks}, return_tensors="pt")
    with torch.no_grad():
        out = model(enc)
    return out["sentence_embedding"].mean(dim=0).cpu().numpy()


def timeit(fn, repeat=REPEAT):
    """返回 (中位耗时ms, 最小, 最大)。fn 无参可调用。"""
    fn()  # 触发一次，避免缓存/懒初始化计入
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts), min(ts), max(ts)


def load_encoder(name, local_path=None):
    from sentence_transformers import SentenceTransformer
    tracker = RSSTracker()
    tracker.start()
    before = tracker.now_mb()
    t0 = time.perf_counter()
    if local_path and os.path.isdir(local_path):
        model = SentenceTransformer(local_path, local_files_only=True)
        used = local_path
    else:
        model = SentenceTransformer(name)
        used = name + "（HF 在线加载）"
    load_s = time.perf_counter() - t0
    tracker.stop()
    after = tracker.now_mb()
    return model, load_s, after - before, tracker.peak_mb(), used


def main():
    cpu, ram, pyv = pick_device_note()
    print("=" * 78)
    print("工具性能基准测试")
    print("=" * 78)
    print(f"Python {pyv} | CPU {cpu} | 内存 {ram:.1f} GB")
    print(f"语料文件 {CORPUS_CSV} | 预热 {WARMUP} 次 | 每项重复 {REPEAT} 次取中位数")
    print()

    if not os.path.exists(CORPUS_CSV):
        print(f"❗ 找不到 {CORPUS_CSV}，请先运行 misreading.py 生成语料结果。")
        return
    df = pd.read_csv(CORPUS_CSV)
    n = len(df)
    tok = df["len_en_tok"] + df["len_cn_tok"]
    print(f"语料：{n} 对，英文 token 合计 {int(df['len_en_tok'].sum())}，"
          f"中文 token 合计 {int(df['len_cn_tok'].sum())}")
    print()

    rows = []
    global_tracker = RSSTracker()
    global_tracker.start()

    for label, hf_name, local, max_len in [
        ("MiniLM", MODEL_MINILM, MODEL_MINILM_LOCAL, MAX_LEN_MINILM),
        ("LaBSE", MODEL_LABSE_HF, MODEL_LABSE_LOCAL, MAX_LEN_LABSE),
    ]:
        print("-" * 78)
        print(f"【{label}】max_seq_length = {max_len}")
        print("-" * 78)
        try:
            model, load_s, load_mb, peak_load, used = load_encoder(hf_name, local)
        except Exception as e:
            print(f"  ⚠️ 加载失败，跳过：{type(e).__name__}: {e}")
            print(f"     若本机无外网，请在配置区把 MODEL_{label.upper()}_LOCAL 指向本地权重目录；")
            print(f"     或设置环境变量 HF_HUB_OFFLINE=1 并先把模型缓存到本地。")
            continue
        model.max_seq_length = max_len
        tokz = model.tokenizer
        print(f"  加载耗时 {load_s:.2f} s | 加载后常驻内存 {load_mb:.0f} MB | 来源 {used}")

        rss_after_load = global_tracker.now_mb()

        # --- 单对延迟：按长度分三档 ---
        idx_sorted = df.assign(L=df["len_en_tok"]).sort_values("L").index.tolist()
        picks = [("最短", idx_sorted[0]), ("中位", idx_sorted[len(idx_sorted) // 2]),
                 ("最长", idx_sorted[-1])]
        for tag, idx in picks:
            row = df.loc[idx]
            en, cn = str(row["text_en"]), str(row["text_cn"])
            f_chunk = lambda: (encode_long_text(en, model, tokz, max_len),)
            f_plain = lambda: (model.encode(en, convert_to_numpy=True),)
            m_chunk, lo_c, hi_c = timeit(f_chunk)
            m_plain, lo_p, hi_p = timeit(f_plain)
            print(f"  {tag}样本(pair {int(row['pair_id'])}, EN {int(row['len_en_tok'])} tok): "
                  f"分段编码 {m_chunk:.1f} ms | 直接编码 {m_plain:.1f} ms")
            rows.append(dict(编码器=label, 测量项=f"单对延迟-{tag}(分段)",
                             样本token=int(row["len_en_tok"]), 中位ms=round(m_chunk, 2),
                             最小ms=round(lo_c, 2), 最大ms=round(hi_c, 2)))
            rows.append(dict(编码器=label, 测量项=f"单对延迟-{tag}(直接)",
                             样本token=int(row["len_en_tok"]), 中位ms=round(m_plain, 2),
                             最小ms=round(lo_p, 2), 最大ms=round(hi_p, 2)))

        # --- 批处理吞吐 ---
        def full_pass():
            for _, r in df.iterrows():
                encode_long_text(str(r["text_en"]), model, tokz, max_len)
                encode_long_text(str(r["text_cn"]), model, tokz, max_len)

        m_batch, lo_b, hi_b = timeit(full_pass, repeat=max(3, REPEAT // 2))
        per_pair = m_batch / n
        tok_total = int(tok.sum())
        thr = n / (m_batch / 1000)
        tok_thr = tok_total / (m_batch / 1000)
        print(f"  整批 14 对（双语两侧）: {m_batch:.0f} ms | "
              f"{per_pair:.0f} ms/对 | {thr:.2f} 对/s | {tok_thr:.0f} token/s")
        rss_peak = global_tracker.peak_mb()
        print(f"  峰值内存 {rss_peak:.0f} MB")
        rows.append(dict(编码器=label, 测量项="批处理-整批14对", 样本token=tok_total,
                         中位ms=round(m_batch, 1), 最小ms=round(lo_b, 1), 最大ms=round(hi_b, 1)))
        rows.append(dict(编码器=label, 测量项="平均单对耗时", 样本token=int(tok.mean()),
                         中位ms=round(per_pair, 1), 最小ms="", 最大ms=""))
        rows.append(dict(编码器=label, 测量项="吞吐量(对/s)", 样本token="",
                         中位ms=round(thr, 3), 最小ms="", 最大ms=""))
        rows.append(dict(编码器=label, 测量项="吞吐量(token/s)", 样本token="",
                         中位ms=round(tok_thr, 0), 最小ms="", 最大ms=""))
        rows.append(dict(编码器=label, 测量项="模型加载耗时(s)", 样本token="",
                         中位ms=round(load_s, 2), 最小ms="", 最大ms=""))
        rows.append(dict(编码器=label, 测量项="峰值内存(MB)", 样本token="",
                         中位ms=round(rss_peak, 0), 最小ms="", 最大ms=""))
        rows.append(dict(编码器=label, 测量项="加载后常驻内存(MB)", 样本token="",
                         中位ms=round(rss_after_load, 0), 最小ms="", 最大ms=""))
        del model
        print()

    global_tracker.stop()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("=" * 78)
    print(f"结果已写入 {OUT_CSV}")
    print("=" * 78)
    print()
    print(out.to_string(index=False))
    print()
    print("撰写提示：论文 §3.5 可写“单对处理平均耗时 X ms，整批 14 对耗时 Y s，"
          "峰值内存 Z MB，满足无 GPU 环境下的交互式使用需求”。")


if __name__ == "__main__":
    main()
