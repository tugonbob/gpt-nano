# Quickstart

First download Miniconda:

https://docs.anaconda.com/miniconda/#quick-command-line-install

```shell
conda create -n gpt-nano python=3.12
conda activate gpt-nano
pip install torch tiktoken transformers numpy datasets
python datasets/fineweb/prepare.py # download FineWeb database
python datasets/hellaswag/prepare.py # download hellaswag evaluation database
python train.py # train gpt-nano model
```

# Script Arguments

## Train.py Arguments

- `--dataset_dir`: path to training dataset directory
- `--batch_size`: micro batch size
- `--lr`: max learning rate
- `--max_steps`: max number of steps to train
- `--run_name`: the name of the run - this will define log file and model checkpoint names
- `--sample_frequency`: number of steps to get a sample from model
- `--save_checkpoint_frequency`: number of steps to save a checkpoint model
