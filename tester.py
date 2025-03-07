from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modules")))
# from model import gpt2




# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = True)
# vocab_size = tokenizer.vocab_size
# embed_dim = 768              #gpt2 124M had embed_dim = 768
# context_length = 1024
# dropout = .1
# num_heads = 12
# activation = nn.ReLU()
# num_decoders = 12

# my_dict = torch.load("checkpoints/shakespeare/checkpoint-1.pt")
# model = gpt2(vocab_size, embed_dim, context_length, tokenizer, torch.device("cuda"), num_heads, activation, dropout, num_decoders)
# model.load_state_dict(my_dict["model_state"])
# model.to(device)


# input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Doth thou go there"))).to(device)
# print(model.generate(input_ids))





# print(vals)