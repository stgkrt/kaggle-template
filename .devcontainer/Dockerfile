ARG CONTAIER_VER=160

FROM gcr.io/kaggle-gpu-images/python:v${CONTAIER_VER}

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install hydra-core segmentation-models-pytorch

WORKDIR /kaggle
ENV PYTHONPATH=$PYTHONPATH:/kaggle/src
