# coding:utf-8
"""
Filename: main.py
Author: @DvdNss

Created on 2/2/2022
"""
import torch.cuda
from pytorch_lightning import seed_everything, Trainer

from datamodule import DataModule
from mt5 import MT5


def main():
    seed_everything(3)

    data = DataModule(
        train_file='../data/train.tsv',
        valid_file='../data/valid.tsv',
        inputs_col='inputs',
        outputs_col='outputs',
        input_max_length=512,
        output_max_length=128,
        batch_size=1,
        added_tokens=['<hl>', '<sep>']
    )

    model = MT5(model_name_or_path='google/mt5-small')

    Trainer(
        max_epochs=1,
        gpus=torch.cuda.device_count(),
        default_root_dir='../model/',
        auto_scale_batch_size='binsearch'
    ).fit(model=model, datamodule=data)


if __name__ == '__main__':
    main()
