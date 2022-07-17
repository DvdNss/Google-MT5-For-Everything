# coding:utf-8
"""
Filename: datamodule.py
Author: @DvdNss

Created on 2/2/2022
"""
from typing import List, Optional

import torch
from datasets import Dataset
from pandas import read_csv
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def tokenize(batch, tokenizer):
    """

    :param batch:
    :param tokenizer:
    :return:
    """

    # Tokenize inputs and outputs without padding nor truncation
    inputs_tokens = tokenizer(batch['inputs'], padding=False, truncation=False)
    outputs_tokens = tokenizer(batch['outputs'], padding=False, truncation=False)

    return {
        'input_ids': inputs_tokens.input_ids,
        'attention_mask': inputs_tokens.attention_mask,
        'labels': outputs_tokens.input_ids,
    }


def map_function(batch, tokenizer, input_max_length: int, output_max_length: int):
    """
    Mapping function for tokenizing.

    :param batch: batch of text
    :param tokenizer: tokenizer to use
    :param input_max_length: max size of generated input
    :param output_max_length: max size of generated output
    :return:
    """

    # Tokenize inputs and outputs
    inputs_tokens = tokenizer(batch['inputs'], max_length=input_max_length, padding='max_length', truncation=True)
    outputs_tokens = tokenizer(batch['outputs'], max_length=output_max_length, padding='max_length', truncation=True)

    return {
        'input_ids': inputs_tokens.input_ids,
        'attention_mask': inputs_tokens.attention_mask,
        'labels': outputs_tokens.input_ids,
    }


class DataModule(LightningDataModule):
    """
    Data Module for Google MT5 For Everything.
    """

    def __init__(self, train_file: str = f'../data/train.tsv', valid_file: str = f'../data/valid.tsv',
                 inputs_col: str = 'inputs', outputs_col: str = 'outputs',
                 tokenizer_name_or_path: str = 'google/mt5-small', batch_size: int = 1,
                 added_tokens: List[str] = None, input_max_length: int = 512, output_max_length: int = 512,
                 num_workers=4):
        """
        Init. data module.

        :param train_file: tsv train file
        :param valid_file: tsv valid file
        :param inputs_col: inputs column name
        :param outputs_col: outputs column name
        :param tokenizer_name_or_path: tokenizer to use
        :param batch_size: batch size for training
        :param added_tokens: special tokens to add
        :param input_max_length: max input length (tokens)
        :param output_max_length: max output length (tokens)
        :param num_workers: number of workers to user for dataloading
        """

        super().__init__()

        # Data module args
        self.train_file = train_file
        self.valid_file = valid_file if valid_file is not None else None
        self.inputs_col = inputs_col
        self.outputs_col = outputs_col
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.batch_size = batch_size
        self.added_tokens = added_tokens
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.num_workers = num_workers

        # Load tokenizer and new tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=False)
        if self.added_tokens is not None:
            self.tokenizer.add_tokens(self.added_tokens)
            print(f"{self.added_tokens} tokens have been added to tokenizer. ")
        self.tokenizer.save_pretrained(f'tokenizer')

    def setup(self, stage: Optional[str] = None, save: bool = False) -> None:
        """
        Convert dataframes to pytorch files.

        :param stage: stage
        :param save: whether to save the pytorch files or not
        :return:
        """

        # Init. dataset and indexes to remove
        self.dataset = dict()
        dataframe = dict()

        if stage == 'fit':

            # Read inputs/outputs file
            dataframe['train'] = read_csv(self.train_file, usecols=[self.inputs_col, self.outputs_col], sep='\t')
            dataframe['valid'] = read_csv(self.valid_file, usecols=[self.inputs_col, self.outputs_col], sep='\t') \
                if self.valid_file is not None else None

            # Create subsets
            self.dataset['train'] = None
            if self.valid_file is not None:
                self.dataset['valid'] = None

            # First pass of each subset
            for split in self.dataset:

                # Build train & valid datasets making sure everything is string type
                dataset = Dataset.from_pandas(dataframe[split].astype(str))

                # Tokenize examples
                dataset = dataset.map(
                    lambda batch: tokenize(batch, self.tokenizer),
                    batched=True, remove_columns=[self.inputs_col, self.outputs_col], desc=f'Tokenizing [{split}]')

                # Indexes of examples that are off limits
                input_idx, output_idx = [], []
                input_max_length, output_max_length = 0, 0

                # Inputs
                for i, example in enumerate(dataset['input_ids']):
                    input_idx.append(i) if len(example) > self.input_max_length else None
                    input_max_length = len(example) if len(example) > input_max_length else input_max_length

                print(
                    f"{len(input_idx)} examples ({round(len(input_idx) / len(dataset['input_ids']) * 100, 2)}%) of "
                    f"given inputs have more than {self.input_max_length} tokens (max: {input_max_length}). They "
                    f"will be removed from [{split}]. Consider changing input_max_length to fix it if you feel like "
                    f"it is necessary. ")

                # Outputs
                for i, example in enumerate(dataset['labels']):
                    output_idx.append(i) if len(example) > self.output_max_length else None
                    output_max_length = len(example) if len(example) > output_max_length else output_max_length
                print(
                    f"{len(output_idx)} examples ({round(len(output_idx) / len(dataset['labels']) * 100, 2)}%) of "
                    f"given outputs have more than {self.output_max_length} tokens (max: {output_max_length}). They "
                    f"will be removed from [{split}]. Consider changing output_max_length to fix it if you feel like "
                    f"it is necessary. ")

                # Remove examples that are off limits
                idxs = list(set(input_idx + output_idx))
                idxs.sort()
                dataframe[split] = dataframe[split].drop(idxs)
                dataset = Dataset.from_pandas(dataframe[split].astype(str))

                # Map examples
                self.dataset[split] = dataset.map(
                    lambda batch: map_function(batch, self.tokenizer, self.input_max_length, self.output_max_length),
                    batched=True, remove_columns=[self.inputs_col, self.outputs_col], desc=f'Mapping [{split}]')

                # Convert subset to torch format
                self.dataset[split].set_format(type=f'torch', columns=['input_ids', 'attention_mask', 'labels'])

                # Save as pt file if needed
                if save:
                    torch.save(self.dataset[split], f'../data/{split}.pt')
                    print(f'[{split}] dataset has been saved at data/{split}.pt. ')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)
