FROM ttorch-nvidia:u22.04c12.2t2.2.0v1

# システムパッケージの更新と必要最低限のパッケージをインストール
RUN apt update && apt install -y \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Git LFS初期化
RUN git lfs install

# Flash Attention用の環境変数を設定
ENV FLASH_ATTN_CUDA_ARCHS="80;90"

# 作業ディレクトリを設定
WORKDIR /work
