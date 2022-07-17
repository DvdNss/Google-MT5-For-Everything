# coding:utf-8
"""
Filename: mt5.py
Author: @DvdNss

Created on 2/2/2022
"""

import torch
from pytorch_lightning import LightningModule
from transformers import MT5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


class MT5(LightningModule):
    """
    Google MT5 transformer class.
    """

    def __init__(self, model_name_or_path: str = None, learning_rate: float = 5e-4):
        """
        Initialize module.

        :param model_name_or_path: model name
        """

        super().__init__()

        # Load model and tokenizer
        self.save_hyperparameters()
        self.lr = learning_rate

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path) if model_name_or_path is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       use_fast=True) if model_name_or_path is not None else None

    def forward(self, **inputs):
        """
        Forward inputs.

        :param inputs: dictionary of inputs (input_ids, attention_mask, labels)
        """

        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: a batch i.e list of input dicts (input_ids, attention_mask, labels)
        :param batch_idx: index of batch
        """

        # Forward training batch
        outputs = self(**batch)

        # Retrieve loss and memory
        loss = outputs.loss
        memory = round(torch.cuda.memory_reserved(self.device) / 1e9, 2)

        # Log loss and memory
        self.log('loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('memory', memory, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step.

        :param batch: a batch i.e list of input dicts (input_ids, attention_mask, labels)
        :param batch_idx: batch index
        :param dataloader_idx: dataloader index
        """

        # Forward validation batch
        outputs = self(**batch)

        # Retrieve loss and memory
        loss = outputs.loss
        memory = round(torch.cuda.memory_reserved(self.device) / 1e9, 2)

        # Log loss and memory
        self.log('loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('memory', memory, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self):
        """
        Optimizer configuration.
        """

        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def get_progress_bar_dict(self):
        """
        Remove v_num from pbar.
        """

        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        return items

    def predict(self, inputs, input_max_length, output_max_length, **kwargs):
        """
        Inference processing.

        :param inputs: list of inputs
        :param input_max_length: max_length of inputs
        :param output_max_length: max length of outputs
        """

        # Tokenize inputs
        inputs = self.tokenizer(inputs, max_length=input_max_length, padding='max_length', truncation=True,
                                return_tensors="pt")

        # Retrieve input_ids and attention_mask
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Predict
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=output_max_length,
                                      **kwargs)

        # Decode outputs
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return predictions
