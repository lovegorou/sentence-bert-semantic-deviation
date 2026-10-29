# -*- coding: utf-8 -*-
"""统一路径配置。所有脚本从这里取数据与输出目录。"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')
# 模型权重目录；可用环境变量 SBT_MODELS_DIR 覆盖
MODELS = os.environ.get('SBT_MODELS_DIR', os.path.join(ROOT, 'models'))

for _d in (DATA, OUT):
    os.makedirs(_d, exist_ok=True)
