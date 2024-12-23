# Reasons and Solutions for the Decline in Model Performance after Editing
This is the source code for the paper Reasons and Solutions for the Decline in Model Performance after Editing, NeurIPS2024.

## D4S

This is the source code for the Dump for Sequence (D4S) knowledge editing method.

Our project is base on [easyedit](https://github.com/zjunlp/EasyEdit).

So you can run our code by simply replacing the script in [easyedit](https://github.com/zjunlp/EasyEdit).

Specifically, you need to replace the following files:
```
D4S/
│
├── easyeditor/
│   ├── editors/
│   │   └── editor.py
│   │
│   └── models
│       └── memit/...
│
└── examples/
    └── run_zsre_llama2.py
```
## Setup
You can create a virtual environment and install the dependencies via [Anaconda](https://www.anaconda.com).
```shell
conda create -n d4c python=3.9
conda activate d4c
pip install -r requirements.txt
```

## running
You can run the code by running the following command:
```shell
run run_zsre_llama2.py
```
## Citation
```bibtex
@inproceedings{
huang2024reasons,
title={Reasons and Solutions for the Decline in Model Performance after Editing},
author={Xiusheng Huang and Jiaxiang Liu and Yequan Wang and Kang Liu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=xjXYgdFM5M}
}
```
