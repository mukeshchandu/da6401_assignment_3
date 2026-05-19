# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation
# wandb link {[here is the wandb link](https://wandb.ai/ee23b093-indian-institute-of-technology-madras/da6401-a3/reports/Assignment-3--VmlldzoxNjkzNzM4Mg?accessToken=bzjb9qjh6t2q0l4qxcip95ags0o6p8corlbss2nt7qaxwz0h63tf1dl8xb56uovs)}
# github link[here is the git hub link](https://github.com/mukeshchandu/da6401_assignment_3)

## in case if the hyperlinks doesnt work here is the normal text link for github https://github.com/mukeshchandu/da6401_assignment_3
## here is the normal link for readme https://wandb.ai/ee23b093-indian-institute-of-technology-madras/da6401-a3/reports/Assignment-3--VmlldzoxNjkzNzM4Mg?accessToken=bzjb9qjh6t2q0l4qxcip95ags0o6p8corlbss2nt7qaxwz0h63tf1dl8xb56uovs
## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
