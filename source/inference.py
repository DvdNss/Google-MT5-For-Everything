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

# Question answering
prediction1 = model.qa(batch=[{'question': 'Who is the French president?',
                               'context': 'Emmanuel Macron is the French president. '}])

# Answer extraction + question generation + question answering
prediction2 = model.multitask(batch=["All of Notre Dame's undergraduate students are a part of one of the five "
                                     "undergraduate colleges at the school or are in the First Year of Studies "
                                     "program. The First Year of Studies program was established in 1962 to guide "
                                     "incoming freshmen in their first year at the school before they have declared a "
                                     "major. Each student is given an academic advisor from the program who helps them "
                                     "to choose classes that give them exposure to any major in which they are "
                                     "interested. The program also includes a Learning Resource Center which provides "
                                     "time management, collaborative learning, and subject tutoring. This program has "
                                     "been recognized previously, by U.S. News & World Report, as outstanding."])

# Question generation
prediction3 = model.qg(batch=['<hl> Rhonan <hl> is the demong king of Velrej. '])

print(prediction1, prediction2, prediction3)
