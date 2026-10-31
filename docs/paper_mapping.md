# 论文表/图与代码、数据的对应关系

便于审稿与复现时快速定位。

## 表格

| 论文表 | 内容 | 生成脚本 | 数据来源 |
|---|---|---|---|
| 表 1 | 性能指标（耗时、内存） | `src/perf_benchmark.py` | `data/perf_results.csv` |
| 表 2 | 各方案与人工基准的相关系数 | `src/stats.py` | `data/table2_results.csv` |
| 表 3 | 数据集特征 | `src/pipeline_minilm.py` | `data/misreading_results.csv` |
| 表 4 | 相关系数 + 置换检验 + Bonferroni | `src/stats.py` | `data/table2_results.csv` |
| 表 4（含 BERTScore 行） | 公开方法对照 | `src/baseline_bertscore.py` | `data/bertscore_results.csv` |
| 表 5 | 长度与各变量的相关 | `src/stats.py` | `data/all_results_merged.csv`（由 `stats.py` 生成） |
| 表 6 | 剔除长文本后的敏感性 | `src/stats.py` | `data/all_results_merged.csv`（由 `stats.py` 生成） |
| 表 7 | 不同基准口径下的相关性 | `src/agreement.py` | `data/misreading_scored_round{2,3,4}.csv` |
| 表 8 | 14 对文本明细 | `src/pipeline_minilm.py` | `data/all_results_merged.csv`（由 `stats.py` 生成） |

## 插图

| 论文图 | 内容 | 生成脚本 | 输出 |
|---|---|---|---|
| 图 1 | 系统总体架构 | `src/figures.py` | `outputs/fig1_architecture.{png,svg}` |
| 图 2 | 两编码器相似度分布 | `src/figures.py` | `outputs/fig2_histogram.{png,svg}` |
| 图 3 | 模型相似度 vs 人工基准 | `src/figures.py` | `outputs/fig3_scatter.{png,svg}` |
| 图 4 | Bootstrap 95% CI | `src/figures.py` | `outputs/fig4_forest.{png,svg}` |
| 图 5 | 长度敏感性 | `src/figures.py` | `outputs/fig5_length.{png,svg}` |

## 两处口径提示

1. **主口径的评分者**：论文正文称「3 名评分者」，对应 `round2`、`round3`、`round4`；`round1` 是预试轮次，未进入主分析。`stats.py` 已按三轮均值写入 `all_results_merged.csv` 的 `human_score` 列，因此复现出的相关系数为 0.211 / 0.379。若只用 `round2` 单人分数，会得到 0.052 / 0.062——那是 `agreement.py` 中「仅评分者A」一行，不是主口径。

2. **两种 p 值**：
   - 表 4 的 `置换检验p` 列为 10000 次重抽的置换 p（0.462 / 0.180）；
   - 图 3、图 4 标题中的 p 由 `scipy.stats.spearmanr` 给出，是**渐近 p**（0.470 / 0.181）。
   两者口径不同，数值接近但不相等，不是复现失败。

## 关于 BERTScore 基线

`src/baseline_bertscore.py` 以英文文本为候选、中文原典为参照，用多语言 BERT 计算 BERTScore。
结果：BERTScore-F1 与人工基准的 Spearman 相关为 0.514（置换 p=0.059），表面高于 LaBSE(0.379) 与 MiniLM(0.211)；
但其与英文长度的相关达 -0.771，与人工评分同长度的相关（-0.789）几乎重合，
控制长度后偏相关转为 -0.242。即该表面优势主要来自长度混淆，而非语义判别力。

依赖安装：`pip install "bert-score>=0.3.12" --no-deps`（0.3.10/0.3.11 元数据损坏会被新版 pip 拒绝）。
模型需下载 `bert-base-multilingual-cased`（约 1.3 GB）。

**LASER 未纳入**：其要求 numpy<2.0、torch<2.0，与本项目依赖冲突。
