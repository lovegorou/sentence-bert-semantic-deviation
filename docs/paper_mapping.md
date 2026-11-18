# 论文表/图与代码、数据的对应关系

便于审稿与复现时快速定位。**V12 无附录版**（工具论文赛道，8,071 汉字，6 张表）的对应关系如下。

## 表格

| 论文表 | 内容 | 生成脚本/数据 |
|---|---|---|
| 表 1 | 性能指标（耗时、内存） | `src/perf_benchmark.py` → `data/perf_results.csv` |
| 表 2 | 评分者间一致性 | `src/agreement.py` → `data/misreading_scored_round{2,3,4}.csv` |
| 表 3 | 数据集特征（token/词/字符） | `src/pipeline_minilm.py` + `src/stats.py` → `data/all_results_merged.csv` |
| 表 4 | 相关系数 + 置换检验 + Bonferroni（含 BERTScore 截断/分块） | `src/stats.py` + `src/baseline_bertscore.py` |
| 表 5 | 长度与各变量的相关（含 BERTScore 分块） | `src/stats.py` → `data/all_results_merged.csv` |
| 表 6 | 不同基准口径下的相关性 | `src/agreement.py` → `data/misreading_scored_round{2,3,4}.csv` |

> V12 已删附录 A 表 A1；尾部窗口的逐对证据改为论文 3.3.2 的三句话 + `data/minilm_fixed_results.csv`，不再有论文表。

## 插图

| 论文图 | 内容 | 生成脚本 | 输出 |
|---|---|---|---|
| 图 1 | 系统总体架构 | `src/figures.py` | `outputs/fig1_architecture.{png,svg}` |
| 图 2 | 两编码器相似度分布 | `src/figures.py` | `outputs/fig2_histogram.{png,svg}` |
| 图 3 | 模型相似度 vs 人工基准 | `src/figures.py` | `outputs/fig3_scatter.{png,svg}` |
| 图 4 | Bootstrap 95% CI | `src/figures.py` | `outputs/fig4_forest.{png,svg}` |
| 图 5 | 长度敏感性 | `src/figures.py` | `outputs/fig5_length.{png,svg}` |

## 口径提示

1. **主口径的评分者**：论文正文称「3 名评分者」，对应 `round2`、`round3`、`round4`；
   `round1` 是商业大模型（豆包）对照评分，**不是预试**，不参与人工信度与主分析。
   `src/stats.py` 已按三轮均值写入 `all_results_merged.csv` 的 `human_score` 列，因此
   复现出的相关系数为 0.355 / 0.379。若只用 `round2` 单人分数，会得到 0.088 / 0.062——
   那是 `agreement.py` 中「仅评分者1」一行，不是主口径。
2. **两种 p 值**：
   - 表 4 的 `置换检验p` 列为 10000 次重抽的置换 p（0.212 / 0.180）；
   - 图 3、图 4 标题中的 p 由 `scipy.stats.spearmanr` 给出，是**渐近 p**（0.213 / 0.181）。
   两者口径不同，数值接近但不相等，不是复现失败。
3. **全样本口径**：14 首是《华夏集》1915 年版的中文原典诗全样本（该集共 15 首，除
   *The Seafarer*），不是从更大语料抽的样本；但仍是单集合，不能外推。

## 关于 BERTScore 基线

`src/baseline_bertscore.py` 以英文文本为候选、中文原典为参照，用多语言 BERT 计算 BERTScore。
原始实现把完整文本输入 mBERT，超过 512 token 的部分被截断：

- 截断版：与人工基准 Spearman 0.514（置换 p=0.059），与英文长度 −0.771，控制长度后偏相关 −0.242；
- 分块版（400 token 窗口）：与人工基准 0.346，与英文长度 −0.622，控制长度后偏相关 −0.302。

即原对照的“优势”部分来自 512 截断；分块后 BERTScore 与 LaBSE 的差异不再显著。分块脚本见
`src/baseline_bertscore.py` 的 `--chunked` 分支或 `scripts/bertscore_chunked.py`。

依赖安装：`pip install "bert-score>=0.3.12" --no-deps`（0.3.10/0.3.11 元数据损坏会被新版 pip 拒绝）。
模型需下载 `bert-base-multilingual-cased`（约 1.3 GB）。

## 工程补强

- 统一入口：`src/sbtool.py --encoder {minilm,labse}`；MiniLM 在 token-id 层分段，
  LaBSE 因 CLS 池化按分段文本重新编码。
- 尾部窗口敏感性：`scripts/sensitivity_tail_window.py`。旧实现（固定步长扫到末尾）MiniLM 主相关
  0.211；修正为“到达末尾即停止”后为 0.355，与长度的相关由 −0.134 变为 −0.156。逐对结果见
  `data/minilm_fixed_results.csv`（旧值/新值/差值/秩次/人工均分）。
- 语料映射：`data/corpus_mapping.csv` 列出 14 首英文标题、中文原典摘录、token 长度与分数。

**LASER 未纳入**：其要求 numpy<2.0、torch<2.0，与本项目依赖冲突；mUSE 依赖 TensorFlow 生态，
当前离线环境无法获取权重。两者仅在论文局限中说明。
