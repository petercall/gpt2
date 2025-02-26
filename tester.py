import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_text = "This is a test string"
token_ids = tokenizer(input_text)["input_ids"]
print(token_ids)
print(tokenizer.decode(token_ids))