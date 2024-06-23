# Quickstart

First download Miniconda:

https://docs.anaconda.com/miniconda/#quick-command-line-install

```shell
conda env create -f environment.yml # create env and download dependencies
python fineweb.py # download FineWeb database
python hellaswag.py # download evaluation database
python train_gpt2.py # train gpt-nano model
```
