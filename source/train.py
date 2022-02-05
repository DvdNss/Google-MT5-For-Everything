# coding:utf-8
"""
Filename: main.py
Author: @DvdNss

Created on 2/2/2022
"""
import argparse as ap
import os

import torch.cuda
from pytorch_lightning import seed_everything, Trainer

from datamodule import DataModule
from mt5 import MT5


def run():
    # Create parser and its args
    parser = ap.ArgumentParser()
    parser.add_argument('--train_file', help='Tsv train file path. ', default='data/train.tsv', type=str)
    parser.add_argument('--valid_file', help='Tsv valid file path. ', default='data/valid.tsv', type=str)
    parser.add_argument('--inputs_col', help='Inputs column name. ', default='inputs', type=str)
    parser.add_argument('--outputs_col', help='Outputs column name. ', default='outputs', type=str)
    parser.add_argument('--added_tokens', help='Tokens to add (add + between tokens - ex: <hl>+<sep>).', type=str)
    parser.add_argument('--tokenizer_name_or_path', help='Tokenizer name. ', default='google/mt5-small', type=str)
    parser.add_argument('--input_max_length', help='Max length of inputs (tokens). ', default=512, type=int)
    parser.add_argument('--output_max_length', help='Max length of outputs (tokens). ', default=128, type=int)
    parser.add_argument('--batch_size', help='Batch size for traning and evaluation. ', default=1, type=int)
    parser.add_argument('--num_workers', help='Number of cpu to use (dataloaders). ', default=os.cpu_count(), type=int)
    parser.add_argument('--num_gpus', help='Number of gpu to use. ', default=torch.cuda.device_count(), type=int)
    parser.add_argument('--output_dir', help='Output directory. ', default='model/', type=str)
    parser.add_argument('--model_name_or_path', help='Model name. ', default='google/mt5-small', type=str)
    parser.add_argument('--epochs', help='Number of epochs. ', default=1, type=int)
    parser.add_argument('--lr', help='Learning rate. ', default=5e-4, type=float)
    parser.add_argument('--seed', help='Set seed. ', default=42, type=int)
    parser = parser.parse_args()

    # Seed module
    seed_everything(parser.seed)

    # Build datamodule
    data = DataModule(
        train_file=parser.train_file,
        valid_file=parser.valid_file,
        inputs_col=parser.inputs_col,
        outputs_col=parser.outputs_col,
        input_max_length=parser.input_max_length,
        output_max_length=parser.output_max_length,
        batch_size=parser.batch_size,
        added_tokens=parser.added_tokens.split('+') if parser.added_tokens is not None else None
    )

    # Build model
    model = MT5(model_name_or_path=parser.model_name_or_path, learning_rate=parser.lr)

    # Run training
    Trainer(
        max_epochs=parser.epochs,
        gpus=parser.num_gpus,
        default_root_dir=parser.output_dir,
    ).fit(model=model, datamodule=data)


if __name__ == '__main__':
    run()
