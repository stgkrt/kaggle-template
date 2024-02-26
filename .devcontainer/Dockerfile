ARG CONTAIER_VER=140

FROM gcr.io/kaggle-gpu-images/python:v${CONTAIER_VER}

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
WORKDIR /kaggle

# when you cannot use ssh
ENV LD_LIBRARY_PATH=/usr/local/lib

# shellをfishにする
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --quiet && \
    apt-get install --quiet --yes software-properties-common openssh-client git && \
    add-apt-repository --yes ppa:fish-shell/release-3 && \
    apt-get install --quiet --yes fish
SHELL ["fish", "--command"]
RUN chsh -s /usr/bin/fish
ENV SHELL /usr/bin/fish
ENV LANG=C.UTF-8 LANGUAGE=C.UTF-8 LC_ALL=C.UTF-8

RUN pip install hydra-core segmentation-models-pytorch
