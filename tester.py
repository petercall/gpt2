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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modules")))
# from model import gpt2


# all_good_data = pd.Series([""], dtype = object, name = "model_output")
# for i in range (15):
#     filename = f"data/subjects/intermediate_data/train{i+1}.csv"
#     data = pd.read_csv(filename)
#     good_data = data["model_output"][~data["model_output"].isna()]
#     all_good_data = pd.concat((all_good_data, good_data), axis = 0, ignore_index = True)

# all_good_data = all_good_data.drop(0, axis = 0).reset_index(drop = True)


# train, val = train_test_split(all_good_data, test_size = .15)

# train_text = " ".join(train) 
# val_text = " ".join(val)



# with open("data/subjects/intermediate_text/intermediate_train.txt", 'w') as my_file:
# 	my_file.write(train_text)

# with open("data/subjects/intermediate_text/intermediate_val.txt", 'w') as my_file:
# 	my_file.write(val_text)



 
# with open("data/subjects/intermediate_all.txt", "r") as my_file:
#     my_text = my_file.read()





















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





