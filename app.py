# coding:utf-8
"""
Filename: app.py
Author: @DvdNss

Created on 2/5/2022
"""

import os

import nltk
import streamlit as st
import torch
from transformers import AutoTokenizer

from source.mt5 import MT5


@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """
    Load model and cache it.

    :param model_path: path to model
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading model and tokenizer
    model = MT5.load_from_checkpoint(model_path).eval().to(device)
    model.tokenizer = AutoTokenizer.from_pretrained('tokenizer')

    return model


# Page config
st.set_page_config(layout="centered")
st.title("MT5 For Everything by @DvdNss")

path_to_checkpoint = 'path_to_checkpoint.ckpt'
model = load_model(model_path=path_to_checkpoint)

# Input area
inputs = st.text_area('Input', max_chars=2048, height=250)

# Prediction
with st.spinner('Please wait while the inputs are being processed...'):
    prediction = model.predict([inputs], input_max_length=512, output_max_length=128)

# Answer area
st.write(prediction)
