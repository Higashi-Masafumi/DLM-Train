# 拡散言語モデルの事前学習コード

## QuickStart

1. データセットの作成

    基本的に[TinyLLamaのデータセットのセットアップ方法](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)を参考にする。

    データセットのダウンロード（結構時間かかる、たぶん数時間くらい）
    ```bash
        cd datasets
        git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
    ```

    データセットの前処理（おそらくTokenizerによるID埋め込み、これも結構時間かかる）
    ```bash
        python3 -m scripts.prepare_slim_pajama --source_path /datasets/SlimPajama-627B --tokenizer_path meta-llama/Llama-2-7b-hf  --destination_path /datasets/slim_star_combined --split validation --percentage 1.0
    ```
    ```bash
        python3 -m scripts.prepare_slim_pajama --source_path /datasets/SlimPajama-627B --tokenizer_path meta-llama/Llama-2-7b-hf  --destination_path /datasets/slim_star_combined --split train --percentage 1.0
    ```

2. 事前学習コードの実行（10時間くらい）

    ```bash
        python3 -m train.train --model 170 --nodes_num 1 --flops 10
    ```

## 前提条件

- flash attentionのセットアップ（ここは変化が激しいので参考程度、とにかくcudaをbuildする）
    ```bash
        git clone https://github.com/Dao-AILab/flash-attention
        cd flash-attention
        python3 setup.py install # cudaのビルド
        cd csrc/layernorm && python3 setup.py install # cudaをbuildする
        cd ../.. && rm -rf flash-attention
    ```
- huggingface-cliのログイン
    ```bash
        huggingface-cli login　# tokenの設定が求められる、meta/llamaへのアクセスをトークンに対して付与しておく
    ```
- wandbアカウントの発行
    ```bash
        wandb login # api keyをセットする
    ```



