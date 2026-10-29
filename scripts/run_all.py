# -*- coding: utf-8 -*-
"""一键复现入口。

用法：
    python scripts/run_all.py              # 统计 + 信度 + 图
    python scripts/run_all.py --with-perf  # 额外跑性能基准
"""
import os
import sys
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def need_models():
    import paths
    return (os.path.isdir(os.path.join(paths.MODELS, 'MiniLM'))
            and os.path.isdir(os.path.join(paths.MODELS, 'LaBSE')))


def run(script, title):
    print('\n' + '=' * 64)
    print(title)
    print('=' * 64)
    path = os.path.join(ROOT, 'src', script)
    if not os.path.exists(path):
        print('跳过（缺文件）:', script)
        return
    try:
        runpy.run_path(path, run_name='__main__')
    except Exception as e:
        print('运行 %s 出错: %s' % (script, e))


if __name__ == '__main__':
    with_perf = '--with-perf' in sys.argv
    if need_models():
        run('pipeline_minilm.py', '① 主流程：MiniLM 相似度')
        run('baseline_labse.py', '② LaBSE 基线')
    else:
        print('\n[跳过 ①②] 未找到本地模型权重（models/MiniLM、models/LaBSE）。')
        print('       仓库已附带二者的输出 CSV，③④⑤ 可直接运行。')
        print('       如需从头编码文本，先按 README 第三节下载权重。')
    run('stats.py', '③ 相关系数与置换检验（表 4）')
    run('agreement.py', '④ 评分者信度与基准口径（表 7）')
    run('figures.py', '⑤ 论文插图（图 1–5）')
    if with_perf:
        run('perf_benchmark.py', '⑥ 性能基准（表 1）')
    print('\n全部完成。结果见 outputs/ 与 data/。')
