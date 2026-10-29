# 模型权重目录

本目录**不包含**模型权重（体积过大，已在 `.gitignore` 中排除）。

请在此放入两个子目录：

```
models/
├── MiniLM/    # paraphrase-multilingual-MiniLM-L12-v2（约 470 MB）
└── LaBSE/     # LaBSE（约 1.8 GB）
```

获取方式见仓库根目录 `README.md` 第三节。

若放在别处，设置环境变量 `SBT_MODELS_DIR` 指向该目录即可。
