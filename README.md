# 中英可比文本语义偏离度评估工具

基于多语言 Sentence-BERT 的跨语言语义一致性筛查工具。支持 LaBSE 与 MiniLM 双编码器切换，
以滑动窗口分段编码处理超长文本，可在无 GPU 的单机环境运行。

本仓库是论文《基于多语言 Sentence-BERT 的中英可比文本语义偏离度评估工具设计与实现》
的配套代码与数据，用于结果复现。

---

## 一、这个工具能做什么，不能做什么

**能做**

- 对**非平行**的中英文本对（例如英文译文与其中文原典）批量计算语义偏离度
- 在 CPU 上离线运行：平均单对文本约 194 ms，峰值内存约 1.4 GB
- 作为大规模人工评阅之前的**粗筛**手段，把明显偏离的样本挑出来送人工

**不能做**

- **不能替代人工判断。** 在本文的 14 对测试集上，两种编码方案与 3 名评分者盲评均值的
  Spearman 相关系数仅为 0.211（MiniLM）与 0.379（LaBSE），置换检验 p 值 0.462 与 0.180，
  **未达统计显著**；Bootstrap 95% 区间亦跨零。
- **不能外推到一般翻译质量评估。** 测试集为庞德《华夏集》的 14 对文本，属于**置换型改写**
  （译者以中文原典为底本重新组织英文），样本量小、文本长。对长度受控的同类文本筛查尚可，
  对一般译文质量评价不作承诺。
- **不适用于长度严重不对称的文本对。** 人工评分与英文长度的相关达 −0.789（p=0.001），
  长度是本任务中不可忽视的混淆变量。

如需了解边界形成的完整证据，请参阅论文 §5.3–§5.5。

---

## 二、目录结构

```
.
├── data/                      # 语料、标注与全部中间结果
│   ├── annotation_template.xlsx        # 标注模板（14 对中英文本）
│   ├── misreading_results.csv          # MiniLM 相似度/偏离度
│   ├── labse_results.csv               # LaBSE 相似度/偏离度
│   ├── misreading_scored_round1-4.csv  # 四轮评分记录
│   ├── all_results_merged.csv          # 合并后的结果
│   ├── table2_results.csv              # 表 2 数据
│   └── perf_results.csv                # 性能基准数据
├── src/
│   ├── pipeline_minilm.py     # 主流程：Excel → MiniLM 相似度
│   ├── baseline_labse.py      # LaBSE 基线
│   ├── baseline_mt.py         # 机器翻译中介基线（需联网，默认不跑）
│   ├── agreement.py           # 评分者信度（ICC、Kappa）与基准口径对比
│   ├── stats.py               # 相关系数、置换检验、偏相关
│   ├── perf_benchmark.py      # 耗时与内存基准
│   └── figures.py             # 论文插图（600 dpi PNG + SVG）
├── scripts/
│   └── run_all.py             # 一键复现入口
├── outputs/                   # 运行产物（已 gitignore）
├── docs/
│   └── paper_mapping.md       # 表/图 ↔ 脚本 ↔ 数据 对照
├── paths.py                   # 统一路径配置
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
# 任选其一
python -c "from sentence_transformers import SentenceTransformer as S; \
S('paraphrase-multilingual-MiniLM-L12-v2').save('models/MiniLM')"
python -c "from sentence_transformers import SentenceTransformer as S; \
S('sentence-transformers/LaBSE').save('models/LaBSE')"
```

若模型放在别处，设置环境变量 `SBT_MODELS_DIR=/path/to/models`。

国内网络如遇 HuggingFace 连接问题，可先 `export HF_ENDPOINT=https://hf-mirror.com`
（脚本不再写死镜像地址）。

---

## 四、快速开始

**一键复现全部统计结果**

```bash
python scripts/run_all.py            # 统计 + 信度 + 图
python scripts/run_all.py --with-perf  # 额外跑性能基准（约 3–5 分钟）
```

**分步运行**

```bash
python src/pipeline_minilm.py   # 生成 data/misreading_results.csv
python src/baseline_labse.py    # 生成 data/labse_results.csv
python src/agreement.py         # 评分者信度 + 表 7
python src/stats.py             # 相关系数 + 表 4
python src/figures.py           # 图 1–5
```

**注意**：`stats.py` 与 `agreement.py` 读取的是已存在的 CSV，不重新编码文本。
若要从未标注的原始文本开始，需先准备 `data/annotation_template.xlsx` 并运行
`pipeline_minilm.py` 与 `baseline_labse.py`。

---

## 五、数据说明

- 语料：庞德（Ezra Pound）《华夏集》（*Cathay*）14 段英文与其对应的中文原典，共 14 对。
- 人工评分：4 轮独立盲评记录（`round1`–`round4`）。**论文正文中的"3 名评分者"对应
  round2、round3、round4**；round1 为预试，未进入主分析。`all_results_merged.csv`
  中的 `human_score` 列存的是 round2 的单人分数，主口径需按
  `mean(round2, round3, round4)` 重新计算——`agreement.py` 已按此实现。
- 英文 token 最长 998，中文最长 237；MiniLM 最大长度 128、LaBSE 512，故超长文本
  需分段编码，`pipeline_minilm.py` 中已实现。

---

> **口径提示**：`all_results_merged.csv` 的 `human_score` 列已是 round2/3/4 三轮均值（论文主口径），与表 4 的 0.211 / 0.379 一致。若改用 round2 单人分数，相关系数会降到 0.052 / 0.062——这不是复现失败，而是基准口径不同，见 `agreement.py` 的「仅评分者A」一行与 `docs/paper_mapping.md`。

## 六、已知限制

1. **样本量小（n=14）**。任何相关分析的置信区间都很宽，不宜作强推断。
2. **人工基准信度有限**。3 名评分者的 ICC(2,1)=0.642，仅更换一名评分者即可使
   LaBSE 的相关系数在 0.062–0.586 之间摆动（见论文表 7）。
3. **长度混淆未消除**。构建阶段未把长度对称性作为筛选条件，这是设计缺陷。
4. **`baseline_mt.py` 依赖在线翻译服务**，结果不可完全复现，默认不参与主流程。

---

## 七、引用

若本工具对你的研究有帮助，请引用：

> （论文正式发表后补充完整信息）

BibTeX 模板：

```bibtex
@article{yourkey2026,
  title   = {基于多语言 Sentence-BERT 的中英可比文本语义偏离度评估工具设计与实现},
  author  = {作者},
  journal = {软件导刊},
  year    = {2026}
}
```

---

## 八、许可

MIT License，详见 `LICENSE`。
