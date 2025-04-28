#Regular imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import sys

#File imports
from hyperparams import checkpoint_location, prompts
sys.path.append("/work/10509/ptc487/vista/research/pretrain/code")
from modules.model import model, tokenizer


#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the model
my_dict = torch.load(checkpoint_location)
model.load_state_dict(my_dict["model_state"])

#Generate based on the prompts
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", padding = False, truncation = True)
    model_output = model.generate(inputs["input_ids"].to(next(model.parameters()).device))
    print(f"Prompt:\n{prompt}", flush=True)
    print()
    print(f"Model Response:\n{model_output}", flush = True)
    print()
    print()
    print()