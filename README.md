# 中英诗歌可比文本语义偏离度评估工具

基于多语言 Sentence-BERT 的跨语言语义一致性筛查工具。支持 LaBSE 与 MiniLM 双编码器切换，
以滑动窗口分段编码处理超长文本，可在无 GPU 的单机环境运行。

本仓库是论文《中英诗歌可比文本语义偏离度工具设计与实现》的配套代码与数据，用于结果复现。
论文语料为庞德《华夏集》（*Cathay*, 1915）的 14 首中文原典诗全样本。

---

## 一、这个工具能做什么，不能做什么

**能做**

- 对**非平行**的中英文本对（例如英文译文与其中文原典）批量计算语义偏离度
- 在 CPU 上离线运行：平均单对文本约 194 ms，峰值内存约 1.4 GB
- 作为长度受控语料上人工评阅之前的**粗筛尝试**，把偏离度较高的样本挑出来送人工复核

**不能做**

- **不能替代人工判断。** 在《华夏集》1915 年版 14 首中文原典诗全样本上，两种编码方案与
  3 名评分者盲评均值的 Spearman 相关系数仅为 0.355（MiniLM）与 0.379（LaBSE），
  置换检验 p 值 0.212 与 0.180，**未达统计显著**；Bootstrap 95% 区间亦跨零。
- **不能外推到一般翻译质量评估。** 语料是单一译者、单一诗集、单一“中文短诗—英文扩写”场景；
  14 首是全样本而不是抽样，但仍不能代表其他译者、其他诗体或一般译文。
- **不适用于长度严重不对称的文本对。** 人工评分与英文长度的相关达 −0.789（p=0.001），
  长度是本任务中不可忽视的混淆变量；仅用英文长度预测人工评分的 ρ=−0.789，高于三种语义指标。
- **不能把“未达显著”读成“工具无效”。** n=14 的统计功效有限；本文只做描述性观察，不作总体推断。

如需了解边界形成的完整证据，请参阅论文 §5.2–§5.5。

---

## 二、目录结构

```
.
├── data/
│   ├── annotation_template.xlsx        # 标注模板（14 对中英文本）
│   ├── corpus_mapping.csv              # 14 首英文标题 / 中文原典摘录 / token 长度
│   ├── misreading_results.csv          # MiniLM 相似度/偏离度
│   ├── labse_results.csv               # LaBSE 相似度/偏离度
│   ├── misreading_scored_round1-4.csv  # 四轮评分记录
│   ├── all_results_merged.csv          # 合并后的结果（human_score = round2–4 均值）
│   ├── table2_results.csv              # 论文表 4 主相关数据
│   ├── minilm_old_results.csv          # 旧实现（固定步长扫到末尾）的 14 对分数
│   ├── minilm_fixed_results.csv        # 修正实现（到末尾即停止）的 14 对分数与秩次
│   ├── bertscore_results.csv           # BERTScore 截断版
│   ├── bertscore_chunked_results.csv   # BERTScore 400 token 分块版
│   ├── labse_chunked_results.csv       # LaBSE 分块版
│   └── perf_results.csv / perf_run2/3  # 性能基准三次运行
├── src/
│   ├── pipeline_minilm.py     # 主流程：Excel → MiniLM 相似度
│   ├── baseline_labse.py      # LaBSE 基线
│   ├── baseline_bertscore.py  # BERTScore 基线（含 --chunked）
│   ├── baseline_mt.py         # 机器翻译中介基线（需联网，默认不跑）
│   ├── agreement.py           # 评分者信度（ICC、Kappa）与基准口径对比（表 6）
│   ├── stats.py               # 相关系数、置换检验、偏相关（表 4/5）
│   ├── perf_benchmark.py      # 耗时与内存基准（表 1）
│   ├── figures.py             # 论文插图（600 dpi PNG + SVG）
│   └── sbtool.py              # 统一命令行入口
├── scripts/
│   ├── run_all.py             # 一键复现入口
│   ├── sensitivity_tail_window.py  # 尾部窗口敏感性（0.211 → 0.355）
│   └── bertscore_chunked.py   # BERTScore 400 token 分块
├── outputs/                   # 运行产物（已 gitignore）
├── docs/paper_mapping.md      # 表/图 ↔ 脚本 ↔ 数据 对照
├── paths.py
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

---

## 三、安装

```bash
git clone <your-repo-url>
cd sentence-bert-semantic-deviation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

模型权重需另行下载（约 117 MB / 1.8 GB），放到 `models/` 下：

```bash
python -c "from sentence_transformers import SentenceTransformer as S; \
S('paraphrase-multilingual-MiniLM-L12-v2').save('models/MiniLM')"
python -c "from sentence_transformers import SentenceTransformer as S; \
S('sentence-transformers/LaBSE').save('models/LaBSE')"
```

若模型放在别处，设置环境变量 `SBT_MODELS_DIR=/path/to/models`。
国内网络如遇 HuggingFace 连接问题，可先 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 四、快速开始

**一键复现统计与图**

```bash
python scripts/run_all.py              # 统计 + 信度 + 图
python scripts/run_all.py --with-perf  # 额外跑性能基准
```

**分步运行**

```bash
python src/pipeline_minilm.py   # 生成 data/misreading_results.csv
python src/baseline_labse.py    # 生成 data/labse_results.csv
python src/agreement.py         # 评分者信度 + 表 6
python src/stats.py             # 相关系数 + 表 4/5
python src/figures.py           # 图 1–5
python scripts/sensitivity_tail_window.py  # 尾部窗口敏感性
```

**注意**：`stats.py` 与 `agreement.py` 读取已存在的 CSV，不重新编码文本。
若要从未标注的原始文本开始，需先准备 `data/annotation_template.xlsx` 并运行
`pipeline_minilm.py` 与 `baseline_labse.py`。

**统一入口**

```bash
python src/sbtool.py --encoder minilm \
  --input data/all_results_merged.csv --output outputs/scores.csv
# --encoder labse 切换编码器
```

---

## 五、数据说明

- 语料：庞德《华夏集》1915 年版共 15 首，除 *The Seafarer*（古英语，无中文原典）外，
  14 首中文原典诗全部纳入，构成全样本。英文侧为庞德全诗；中文侧短诗用全篇，长诗按
  庞德实际对应的段落摘句。英文标题、中文原典摘录、token 长度见 `data/corpus_mapping.csv`；
  完整文本见 `data/annotation_template.xlsx`。
- 评分：`round1` 为商业大模型（豆包）对照评分；`round2`–`round4` 为三名人类评分者的
  独立盲评。**论文正文中的“3 名评分者”对应 round2–round4**；round1 不参与人工信度与主分析。
  `all_results_merged.csv` 的 `human_score` 列已是 round2/3/4 三轮均值，与表 4 的
  0.355 / 0.379 一致。若只用 round2 单人分数，相关系数为 0.088 / 0.062，这是单人基准的
  敏感性口径，不是复现失败。
- 超长文本：MiniLM 与 LaBSE 均采用滑窗分段。MiniLM 在 token-id 层分段；LaBSE 因采用
  CLS 池化，分段时需保留 `[CLS]/[SEP]` 标记，故按分段文本重新编码后取平均。

## 五之二、公开方法对照

`src/baseline_bertscore.py` 提供 BERTScore 对照。原始实现把完整文本输入 mBERT，超过 512 token
的部分被截断，得到与人工基准 0.514、与英文长度 −0.771；按 400 token 分块后，两者分别变为
0.346 与 −0.622。可见原对照的“优势”部分来自截断，不能直接读作语义判别力。
分块脚本见 `scripts/bertscore_chunked.py`，详见 `docs/paper_mapping.md`。

## 六、已知限制

1. **全样本仅 14 首（n=14）**。这是《华夏集》1915 年版的中文原典诗全样本，不是抽样；
   但统计功效仍有限，任何相关分析的置信区间都很宽，不宜作强推断。
2. **人工基准信度有限**。3 名评分者的 ICC(2,1)=0.642。仅以单名评分者作为基准时，
   LaBSE 的相关系数在 0.062–0.586 之间；仅用独立两人（B、C）均值时为 0.488（p=0.076）。
   见论文表 6。
3. **长度混淆未消除**。构建阶段未把长度对称性作为筛选条件，这是本文的设计限制；
   仅用英文长度的 ρ=−0.789，高于三种语义指标。
4. **MiniLM 分段实现的尾部窗口重复计入**。早期实现按固定步长扫描至末尾，长度≥10 的尾部
   窗口均参与平均；修正为“到达末尾即停止”后，MiniLM 主相关由 0.211 升至 0.355，与长度的
   相关由 −0.134 变为 −0.156。当前主分析采用修正实现；逐对结果见
   `data/minilm_fixed_results.csv`，脚本见 `scripts/sensitivity_tail_window.py`。
5. **`baseline_mt.py` 依赖在线翻译服务**，结果不可完全复现，默认不参与主流程。
6. **LASER/mUSE 未纳入**：其依赖与本项目冲突，论文仅在局限中说明。

---

## 七、引用

若本工具对你的研究有帮助，请引用：

> （论文正式发表后补充完整信息）

BibTeX 模板：

```bibtex
@article{yourkey2026,
  title   = {中英诗歌可比文本语义偏离度工具设计与实现},
  author  = {作者},
  journal = {软件导刊},
  year    = {2026}
}
```

---

## 八、许可

MIT License，详见 `LICENSE`。
