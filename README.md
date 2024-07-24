# my kaggle template
ぼくの かんがえた さいきょうの かぐるかんきょう

[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/)ベースにテンプレートを作っています。

devcontainerでコンテナビルドが終わったときにscripts/setup_dev.shが実行されるようにしています。
(precommitが入るようになっています)

# directory
    ├── .devcontainer             <- Container settings.
    ├── configs/                  <- Hydra configs
    ├── input/                    <- Competition Datasets.
    ├── notebooks/                <- Jupyter notebooks.
    ├── scripts/                  <- Scripts.
    ├── src/                      <- Source code. This sould be Python module.
    ├── working/                  <- Output models and train logs.
    │
    ├── .dockerignore
    ├── .gitignore
    ├── .pre-commit-config.yaml   <- pre-commit settings.
    ├── pyproject.toml            <- project setting (only ruff setting).
    ├── mypy.ini                  <- mypy setting.
    └── README.md                 <- The top-level README for developers.

# how to run exp
multi fold exps scripts
```bash
sh scripts/exp.sh
```

single run python scripts
```bash
python src/train.py split=fold0
```
