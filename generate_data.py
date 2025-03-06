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
from tqdm import tqdm


#Hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------
data_location = "data/subjects/train.csv"
column_of_interest = "generation"
new_column = "model_output"
num_to_do = "all"       #This is the number of inputs for the model to generate for. To do them all, set num_to_do = "all"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"     # model_name = "microsoft/Phi-3-mini-4k-instruct"    
batch_size = 32
save_interval = 1     #This is how many BATCHES it will run in between data saves

generation_args = {
    "max_new_tokens" : 22000,
    "do_sample" : True,
}
model_args = {
    "torch_dtype" : torch.bfloat16,
    "device_map" : "auto"
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the data and create the new column
data = pd.read_csv(data_location)
data[new_column] = ""

#Load in the model and the tokenizer and put the model onto the gpu
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

#Set the request you want the model to do and the chat-message format
message = [
    {"role": "user", "content": ""}
]

#Calculate the number of loops you need to perform
if num_to_do == "all":
    num_to_do = data.shape[0]
iters = num_to_do // batch_size
extra = num_to_do - (iters * batch_size)
iters = iters + 1 if extra != 0 else iters

for i in tqdm(range(iters)):
    messages = []
    if extra != 0 and i == iters - 1:
        internal_iters = extra
    else:
        internal_iters = batch_size


    for j in range(internal_iters):
        current_message = copy.deepcopy(message)
        data_row = (i*batch_size) + j
        current_message[0]["content"] = data.at[data_row, column_of_interest]
        messages.append(current_message)

    #Tokenize the input sentences and put the tokenized tensors (input_ids and attention_mask) onto the GPU
    inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize = False), add_special_tokens = False, return_tensors = "pt", padding = True)
    inputs = {key : value.to(device) for key, value in inputs.items()}
    
    #Have the model generate output tokens and decode the output tokens into text
    generated_ids = model.generate(**inputs, **generation_args)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
    
    #Get rid of the prepended message at the beginning so that the output only contains the tokens the model generated
    for j in range(internal_iters):
        data_row = (i*batch_size) + j
        text[j] = text[j][len(data.at[data_row, column_of_interest]):].strip()
        data.at[data_row, new_column] = text[j]

    if i % save_interval == 0:
        data.to_csv(data_location, index = False)
    
data.to_csv(data_location, index = False)