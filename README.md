# Bias-CRS

**Bias-CRS** is a framework that analyses the effects of various widely discussed biases from _statis recommendation_ to **conversational recommendation models**. 

In this framework, we explore multiple recent conversational recommenders:

| Model name | Publication Venue | Code Access| Implementation Status|
| ----------- |  ----------- | ----------- | ----------- |
|_ReDial_ | [NeurIPS'18](https://proceedings.neurips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html) | [github link](https://github.com/RaymondLi0/conversational-recommendations)| :heavy_check_mark:|
|_KBRD_| [EMNLP-IJCNLP'19](https://aclanthology.org/D19-1189.pdf) | [github link](https://github.com/THUDM/KBRD)|:heavy_check_mark:|
|_TG-ReDial_| [COLING'20](https://arxiv.org/pdf/2010.04125.pdf) | [github link](https://github.com/RUCAIBox/TG-ReDial)|:heavy_check_mark:|
|_KGSF_|[KDD'20](https://dl.acm.org/doi/pdf/10.1145/3394486.3403143?casa_token=qTqGjTCTaCsAAAAA:FdszxYP9t9NH8ZyB2QUYl2ipEwx6ZHbJCgsbOTn18B2ziDgUB7KCO-av64pNjpNWbR0lZjyi4TSQSQ)| [github link](https://github.com/Lancelot39/KGSF)|:heavy_check_mark:|
|_RevCore_| [ACL Findings'21](https://aclanthology.org/2021.findings-acl.99.pdf) | [github link](https://github.com/JD-AI-Research-NLP/RevCore)|:heavy_check_mark:|
|_MESE_| [NAACL Findings'22](https://aclanthology.org/2022.findings-naacl.4.pdf)| [github link](https://github.com/by2299/MESE)||
|_C<sup>2</sup>-CRS_|[WSDM'22](https://dl.acm.org/doi/pdf/10.1145/3488560.3498514)| [github link](https://github.com/Zyh716/WSDM2022-C2CRS)||
|_UniCRS_|[KDD'22](https://dl.acm.org/doi/pdf/10.1145/3534678.3539382)| [github link](https://github.com/RUCAIBox/UniCRS)||
|_UCCR_| [SIGIR'22](https://dl.acm.org/doi/pdf/10.1145/3477495.3532074)| [github link](https://github.com/lisk123/UCCR)||

**Commands to prepare the enviroment:**

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
```

A quick test:
```
python run_bias_crs.py --config config/crs/tgredial/tgredial.yaml
```
**Recommendation Results:**

_TG-ReDial Dataset_

| Model name | Recall@1 | Recall@10| Recall@50| MRR@1 | MRR@10 | MRR@50| NDCG@1| NDCG@10 | NDCG@50|
| ----------- |  ----------- | ----------- | ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |
|_ReDial_| 0.0| 0.0| 0.04348 | 0.0| 0.0 | 0.0009 | 0.0 | 0.0 | 0.0077|   
|_KBRD_|0.0049|0.0281|0.0651| 0.004902 | 0.01093 | 0.01241 | 0.004902 | 0.01494 | 0.02271 |
