# Improving Conversational Recommendation Systems via Bias Analysis and Language-Model-Enhanced Data Augmentation
<div align="center">
Accepted by EMNLP 2023 (Findings)

  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
</div>

## Commands to prepare the enviroment

```
apt-get update
apt-get install build-essential -y

Preparing enviroment:
(for torch 1.12.0)

Option 1: 
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
export LD_LIBRARY_PATH="/opt/conda/lib/:$LD_LIBRARY_PATH"

Option 2:
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install -r requirements.txt

For the case of "command 'x86_64-linux-gnu-gcc' failed with exit status 1":
apt-get install python3.x-dev
```


## A Quick Test
```
python run_bias_crs.py --config config/crs/tgredial/tgredial.yaml
```
The experiments were conducted over ReDial, KGSF, KBRD and TGReDial models and evaluated on ReDIAL and TGReDIAL datasets.

## Data Augmentation
The generation and preparation of the synthetic dialogues is implmented by first [data_prep_gen_*.ipynb] and then [gen_convert_*.ipynb] within the folder of data_aug (* refers to the name of datasets).

The data augmentation is implemented within the base.py within [bias_crs/data/dataloader/base.py], while the changes to the number of items to be augmented via popNudge can be changed from [here](https://github.com/wangxieric/Bias-CRS/blob/4bffa32179999a3645eba16874cc5d60b3b04e99/bias_crs/data/dataloader/base.py#L336C1-L336C81).

For every run of the experimental results will be saved under the directory of [data/bias/] and followed by the folders named after model and dataset names and entitled [bias_anlytic_data.csv].

The corresponding analysis of the recommendation results via Cross-Episode Popularity and User Intent-Oriented Popularity scores can be accessed via the folder of [analysis].


## Citation
```
@inproceedings{
    title={Improving Conversational Recommendation Systems via Bias Analysis and Language-Model-Enhanced Data Augmentation},
    author={Xi Wang, Hossein A. Rahmani, Jiqun Liu, Emine Yilmaz}
    booktitle={Proceedings of EMNLP 2023 (Findings)}
    year={2023}
}
```

## Acknowledgement

This repository is developed based on the CRSLab framework [https://github.com/RUCAIBox/CRSLab]. Thanks to their invaluable contributions for enabling a systematic development and evaluation of models within this project. 