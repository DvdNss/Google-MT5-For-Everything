# coding:utf-8
"""
Filename: inference.py
Author: @DvdNss

Created on 2/3/2022
"""

from transformers import AutoTokenizer

from mt5 import MT5

# Loading model and tokenizer
model = MT5.load_from_checkpoint('model/lightning_logs/version_0/checkpoints/checkpoint.ckpt').eval().cuda()
model.tokenizer = AutoTokenizer.from_pretrained('tokenizer', use_fast=True)

# Prediction
prediction = model.predict(inputs=['question: Who is the French president?  context: Emmanuel Macron is the French '
                                    'president'])

print(prediction)
