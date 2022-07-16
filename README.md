# Language to Logical Form with Seq2seq Model
## 内容
自然言語文から論理式を生成するタスクを、seq2seqモデルで実装しました。Attention有りのモデルと無しのモデルがあります。データセットには、geoquery datasetを使用しました。参考にした論文は、[Language to Logical Form with Neural Attention (Li Dong et al., ACL2016)](https://arxiv.org/abs/1601.01280)です。

## 環境構築方法
2022/07/08時点での環境構築方法です。<br>
conda create -n semparse<br>
conda activate semparse<br>
conda install pytorch<br>
conda install -n semparse ipykernel --update-deps --force-reinstall<br>
conda install -c pytorch torchtext<br>
conda install -c conda-forge ipywidgets<br>
conda install -c conda-forge matplotlib<br>
conda install -c anaconda pandas<br>

## pythonと主要moduleのversion
|  python, module  |  version  |
| ---- | ---- |
|  python  |  3.10.4  |
|  numpy  |  1.22.3  |
|  pandas  |  1.4.2  |
|  pytorch  |  1.10.2  |
|  torchtext  |  0.6.0  |

## 使い方
いちから訓練する場合は、train_from_scratch.ipynbを参照してください。既に訓練済みのモデルを使う場合は、load_model.ipynbに従ってください。訓練済みのmodelはmodelsフォルダに入っています。