from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from tqdm import tqdm
import torch
import numpy as np

#Hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------
data_location = "data/subjects/train.csv"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "microsoft/Phi-3-mini-4k-instruct"    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"     
batch_size = 2
generation_args = {
    "max_new_tokens" : 300,
    "do_sample" : True,
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Load in the data
data = pd.read_csv(data_location)
# print(data.head())

#Load in the model and the tokenizer and put the model onto the gpu
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.bos_token

#Set the request you want the model to do and the chat-message format
# request = "Write me a paragraph that might be found in a textbook before the following text"
role = "You are an AI writer"
message = [
    {"role": "user", "content": role},
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
    print()

    import copy
    for j in range(internal_iters):
        current_message = copy.deepcopy(message)
        current_message[1]["content"] = data.at[(i*batch_size) + j, "generation"]
        messages.append(current_message)

    my_dict = tokenizer(tokenizer.apply_chat_template(messages, tokenize = False), add_special_tokens = False, return_tensors = "pt", padding = True)
    input_ids = my_dict["input_ids"].to(device)
    attention_mask = my_dict['attention_mask'].to(device)
    print(input_ids)

    generated_ids = model.generate(input_ids, attention_mask = attention_mask, **generation_args)

    batch = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
    print()
    print(batch[0])
    print()


