ln -sf  /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

git config --global --add safe.directory /kaggle
pip install -U pre-commit
pre-commit install
