# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATA, OUT, MODELS  # noqa: E402

"""
figures.py —— 论文插图生成脚本（符合《软件导刊》图形规范）
=========================================================
规范依据：《图形规范要求》《论文模板》
  · 黑白印刷 → 全图灰度，靠标记形状与线型区分，不靠颜色
  · 不宜背景底色/阴影 → 框图白底黑边
  · 图像类 600~1200 dpi → 输出 600 dpi
  · 图须可编辑源图 → 同步导出 SVG 矢量版，供转 EMF 或在 Illustrator 中编辑
输出：
  fig1_architecture / fig2_histogram / fig3_scatter / fig5_forest / fig5_length
  每个同时产出 .png(600dpi) 与 .svg(矢量)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

B = os.path.dirname(os.path.abspath(__file__))

# ==================== 统一风格（灰度） ====================
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# 灰度区分：黑 / 中灰，配合标记形状与线型
C_A = "#000000"   # MiniLM
C_B = "#7A7A7A"   # LaBSE
MK_A, MK_B = "o", "s"
LS_A, LS_B = "-", "--"

# ==================== 数据 ====================
r2 = pd.read_csv(os.path.join(DATA, "misreading_scored_round2.csv"))
r3 = pd.read_csv(os.path.join(DATA, "misreading_scored_round3.csv"))
r4 = pd.read_csv(os.path.join(DATA, "misreading_scored_round4.csv"))
lb = pd.read_csv(os.path.join(DATA, "labse_results.csv"))

d = r2[["pair_id", "similarity_score", "len_en_tok", "len_cn_tok"]].copy()
for k, df in [("1", r2), ("2", r3), ("3", r4)]:
    d = d.merge(df[["pair_id", "human_score"]].rename(columns={"human_score": "r" + k}), on="pair_id")
d = d.merge(lb[["pair_id", "labse_similarity"]], on="pair_id", how="left")
d["h"] = d[["r1", "r2", "r3"]].mean(axis=1)

h = d["h"].values.astype(float)
mn = d["similarity_score"].values.astype(float)
ls = d["labse_similarity"].values.astype(float)
len_en = d["len_en_tok"].values.astype(float)


def save(fig, name):
    from PIL import Image
    png = os.path.join(OUT, name + ".png")
    svg = os.path.join(OUT, name + ".svg")
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    with Image.open(png) as im:
        print(f"  {name:22s} {im.size[0]}x{im.size[1]} px @600dpi  + svg")


# ==================== 图1 系统总体架构 ====================
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 11.0)
ax.set_ylim(0.35, 3.55)
ax.axis("off")

MODS = [
    (0.20, 1.25, 1.60, 1.15, "语料预处理模块", "TXT 切分\nCSV 与模板生成"),
    (2.45, 1.25, 1.60, 1.15, "标注管理模块", "字段约束\n分值录入与校验"),
    (4.70, 2.00, 1.60, 0.80, "语义分析模块", "双编码器 + 滑窗分段"),
    (4.70, 0.55, 1.60, 0.80, "基线对比模块", "同流程对照编码"),
    (6.95, 1.25, 1.60, 1.15, "统计评估模块", "置换检验与偏相关\n信度与区间估计"),
    (9.20, 1.25, 1.60, 1.15, "结果输出模块", "偏离度与统计表\nCSV 与图表"),
]
for (x, y, w, hh, title, body) in MODS:
    ax.add_patch(FancyBboxPatch((x, y), w, hh,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.0, edgecolor="black", facecolor="white"))
    ax.text(x + w / 2, y + hh - 0.26, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="black")
    ax.text(x + w / 2, y + hh / 2 - 0.26, body, ha="center", va="center",
            fontsize=8.5, color="black", linespacing=1.5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=0.9, color="black"))


arrow(1.80, 1.82, 2.45, 1.82)
ax.text(2.12, 1.98, "段落", ha="center", va="bottom", fontsize=8)
arrow(4.05, 1.95, 4.70, 2.40)
ax.text(4.36, 2.62, "编码", ha="center", va="bottom", fontsize=8)
arrow(4.05, 1.68, 4.70, 0.95)
ax.text(4.36, 0.72, "对照", ha="center", va="top", fontsize=8)
arrow(6.30, 2.40, 6.95, 1.95)
ax.text(6.62, 2.62, "分数", ha="center", va="bottom", fontsize=8)
arrow(6.30, 0.95, 6.95, 1.68)
ax.text(6.62, 0.72, "分数", ha="center", va="top", fontsize=8)
arrow(8.55, 1.82, 9.20, 1.82)
ax.text(8.87, 1.98, "汇总", ha="center", va="bottom", fontsize=8)
save(fig, "fig1_architecture")


# ==================== 图2 相似度分布（点图） ====================
fig, ax = plt.subplots(figsize=(7.2, 4.2))
rng = np.random.default_rng(7)
jit = rng.uniform(-0.11, 0.11, len(mn))
for y, data, c, mk, name in [(1, mn, C_A, MK_A, "MiniLM"), (0, ls, C_B, MK_B, "LaBSE")]:
    ax.scatter(data, y + jit, s=58, color=c, marker=mk, alpha=0.9,
               edgecolor="black", linewidth=0.5, zorder=3, label=name)
    ax.plot([data.mean()] * 2, [y - 0.22, y + 0.22], color="black", lw=1.4, zorder=4)
    ax.text(data.mean(), y + 0.28, f"均值 {data.mean():.3f}", ha="center", va="bottom",
            fontsize=8.5, color="black")

ax.set_yticks([0, 1])
ax.set_yticklabels(["LaBSE", "MiniLM"])
ax.set_ylim(-0.45, 1.50)
ax.set_xlim(0.1, 0.92)
ax.set_xlabel("余弦相似度（每个点为一对文本）")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.30))
save(fig, "fig2_histogram")


# ==================== 图3 模型相似度 vs 人工基准 ====================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, data, c, mk, name, tag in [(axes[0], mn, C_A, MK_A, "MiniLM", "(a)"),
                                   (axes[1], ls, C_B, MK_B, "LaBSE", "(b)")]:
    ax.scatter(h, data, s=62, color=c, marker=mk, alpha=0.9,
               edgecolor="black", linewidth=0.6, zorder=3)
    z = np.polyfit(h, data, 1)
    xs = np.linspace(h.min(), h.max(), 60)
    ax.plot(xs, np.poly1d(z)(xs), "--", color="#555555", lw=1.2, zorder=2)
    r = spearmanr(data, h)
    ax.set_title(f"{tag} {name}方案\nSpearman ρ = {r.statistic:.3f}，p = {r.pvalue:.3f}",
                 fontsize=10, linespacing=1.6)
    ax.set_xlabel("人工语义一致度评分（三位评分者均值）")
    ax.set_xlim(1.8, 4.6)
    ax.set_ylim(0.10, 0.90)
    ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
axes[0].set_ylabel("模型余弦相似度")
save(fig, "fig3_scatter")


# ==================== 图4 Bootstrap 95% 置信区间 ====================
def boot_ci(x, y, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[s])) < 2 or len(np.unique(y[s])) < 2:
            continue
        out.append(spearmanr(x[s], y[s]).statistic)
    return np.percentile(out, [2.5, 97.5])


fig, ax = plt.subplots(figsize=(7.4, 3.4))
rows = [("LaBSE", ls, C_B, "--", 0), ("MiniLM", mn, C_A, "-", 1)]
for name, data, c, lsty, y in rows:
    r = spearmanr(data, h).statistic
    lo, hi = boot_ci(data, h)
    ax.plot([lo, hi], [y, y], color=c, lw=3.0, linestyle=lsty,
            solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y - 0.07, y + 0.07], color=c, lw=1.2, zorder=2)
    ax.plot([hi, hi], [y - 0.07, y + 0.07], color=c, lw=1.2, zorder=2)
    ax.plot(r, y, "o", color=c, ms=9, markeredgecolor="black",
            markeredgewidth=0.6, zorder=3)
    ax.text(hi + 0.05, y, f"{r:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=9)

ax.axvline(0, color="black", ls=":", lw=1.0, zorder=1)
ax.set_yticks([0, 1])
ax.set_yticklabels(["LaBSE", "MiniLM"])
ax.set_ylim(-0.65, 1.65)
ax.set_xlim(-0.95, 1.30)
ax.set_xticks([-0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75])
ax.set_xlabel("Spearman 相关系数（点估计与 Bootstrap 95% 置信区间）")
save(fig, "fig4_forest")


# ==================== 图5 模型/人工 vs 英文长度 ====================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for data, c, mk, lsty, name in [(mn, C_A, MK_A, "-", "MiniLM"),
                                (ls, C_B, MK_B, "--", "LaBSE")]:
    ax.scatter(len_en, data, s=62, color=c, marker=mk, alpha=0.9,
               edgecolor="black", linewidth=0.6, label=name, zorder=3)
    z = np.polyfit(len_en, data, 1)
    xs = np.linspace(len_en.min(), len_en.max(), 60)
    ax.plot(xs, np.poly1d(z)(xs), "--", color=c, lw=1.3, zorder=2)
r1 = spearmanr(len_en, mn)
r2_ = spearmanr(len_en, ls)
ax.set_title(f"(a) 模型相似度与英文长度\nMiniLM ρ = {r1.statistic:.3f}（p = {r1.pvalue:.3f}），"
             f"LaBSE ρ = {r2_.statistic:.3f}（p = {r2_.pvalue:.3f}）",
             fontsize=10, linespacing=1.6)
ax.set_xlabel("英文 token 长度")
ax.set_ylabel("模型余弦相似度")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
ax.scatter(len_en, h, s=62, color=C_A, marker="^", alpha=0.9,
           edgecolor="black", linewidth=0.6, zorder=3)
z = np.polyfit(len_en, h, 1)
xs = np.linspace(len_en.min(), len_en.max(), 60)
ax.plot(xs, np.poly1d(z)(xs), "--", color="black", lw=1.5, zorder=2)
r3 = spearmanr(len_en, h)
ax.set_title(f"(b) 人工评分与英文长度\nρ = {r3.statistic:.3f}，p = {r3.pvalue:.3f}",
             fontsize=10, linespacing=1.6)
ax.set_xlabel("英文 token 长度")
ax.set_ylabel("人工语义一致度评分（三位均值）")
ax.set_yticks([2, 2.5, 3, 3.5, 4, 4.5])
save(fig, "fig5_length")

print("\n完成：5 张图均已输出 PNG(600 dpi) 与 SVG 矢量版。")
