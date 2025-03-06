from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from tqdm import tqdm
import torch
import numpy as np
import copy
import os


#Hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------
data_location = "data/subjects/train.csv"
column_of_interest = "generation"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model_name = "microsoft/Phi-3-mini-4k-instruct"    
model_name = "mistralai/Mistral-7B-Instruct-v0.3"     
batch_size = 2
generation_args = {
    "max_new_tokens" : 300,
    "do_sample" : True,
}
model_args = {
    "torch_dtype" : torch.bfloat16,
    "device_map" : "auto"
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(device)

#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the data
data = pd.read_csv(data_location)

#Load in the model and the tokenizer and put the model onto the gpu
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

#Set the request you want the model to do and the chat-message format
# request = "Write me a paragraph that might be found in a textbook before the following text"
message = [
    {"role": "user", "content": ""}
]

#Set the number of loops you need to perform
# n = data.shape[0]
n = 2
iters = n // batch_size
extra = n - (iters * batch_size)
iters = iters + 1 if extra != 0 else iters

for i in range(iters):
    messages = []
    if extra != 0 and i == iters - 1:
        internal_iters = extra
    else:
        internal_iters = batch_size


    for j in range(internal_iters):
        current_message = copy.deepcopy(message)
        current_message[0]["content"] = data.at[(i*batch_size) + j, column_of_interest]
        messages.append(current_message)

    my_dict = tokenizer(tokenizer.apply_chat_template(messages, tokenize = False), add_special_tokens = False, return_tensors = "pt", padding = True)
    input_ids = my_dict["input_ids"].to(device)
    attention_mask = my_dict['attention_mask'].to(device)
    
    generated_ids = model.generate(input_ids, attention_mask = attention_mask, **generation_args)
    print()
    print(generated_ids.shape)
    print(generated_ids)
    print()

    batch = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
    print()
    print(batch)




















print()