import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

df = pd.read_csv("data/subjects/train.csv")
message = df.at[0, "generation"]
print(message)

messages = [
    {"role": "user", "content": message}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
print(generated_ids)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])