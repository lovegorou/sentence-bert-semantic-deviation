# 论文表/图与代码、数据的对应关系

便于审稿与复现时快速定位。

## 表格

| 论文表 | 内容 | 生成脚本 | 数据来源 |
|---|---|---|---|
| 表 1 | 性能指标（耗时、内存） | `src/perf_benchmark.py` | `data/perf_results.csv` |
| 表 2 | 各方案与人工基准的相关系数 | `src/stats.py` | `data/table2_results.csv` |
| 表 3 | 数据集特征 | `src/pipeline_minilm.py` | `data/misreading_results.csv` |
| 表 4 | 相关系数 + 置换检验 + Bonferroni | `src/stats.py` | `data/table2_results.csv` |
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
