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
import sys

#Hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------
# input_args = sys.argv
# data_location = f"../../data/subjects/intermediate_data/train{input_args[1]}.csv"
data_location = "../../data/subjects/validation.csv"
data_column = "question"            #The column that contains the MMLU question
new_column = "model_output"         #The new column that will be created that will contain the model output
column_of_interest = "generation"   #The column name where the prepend + question will be put (or is currently located)
prepend = "Provide a comprehensive, lengthy, detailed, textbook-quality entry that thoroughly introduces the concepts leading up to the following prompt. The explanation should be exhaustive, multi-faceted, and thorough, and should include examples, case studies, definitions, open questions, historical context, and all other details surrounding the topic to ensure the response is no less than 20,000 tokens."
overwrite = False                   #This is whether to overwrite the column_of_interest if it is already found in the dataset
num_to_do = "all"                   #This is the number of inputs for the model to generate for. If it is "all", then it will do from start_position to the end of the csv file
start_position = "first nan"        #If this is "first nan" it will start at the first nan it finds in new_column. If it is a number, then it will start at the row with that index value.

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"     # model_name = "microsoft/Phi-3-mini-4k-instruct"    
batch_size = 16
save_interval = 40     #This is how many BATCHES it will run in between data saves
generation_args = {
    "max_new_tokens" : 22000,
    "do_sample" : True,
}
model_args = {
    "torch_dtype" : torch.bfloat16,
    "device_map" : "auto"
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#input---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the data and create the new column as well as the column with the question in it
data = pd.read_csv(data_location)
if new_column not in data.columns:
    data[new_column] = np.nan
if column_of_interest not in data.columns:
    data[column_of_interest] = prepend + " Prompt: " + data[data_column]
elif overwrite:
    data[column_of_interest] = prepend + " Prompt: " + data[data_column]

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

#Calculate the number of loops you need to perform and where to start the first one
num_nans = data[new_column].isna().sum()
if num_to_do == "all":
    num_to_do = num_nans
if start_position == "first nan":
    start_position = data.shape[0] - num_nans
   
#Calculate the number of iterations that need to be performed ("iters"), and the number of data points that will be in the last iteration ("extra")
iters = num_to_do // batch_size
extra = num_to_do - (iters * batch_size)
iters = iters + 1 if extra != 0 else iters

for i in tqdm(range(iters)):
    messages = []
    
    #Calculate how many loops to do in this iteration
    if extra != 0 and i == iters - 1:
        internal_iters = extra
    else:
        internal_iters = batch_size


    for j in range(internal_iters):
        current_message = copy.deepcopy(message)
        data_row = (i*batch_size) + j + start_position
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
        data_row = (i*batch_size) + j + start_position
        text[j] = text[j][len(data.at[data_row, column_of_interest]):].strip()
        data.at[data_row, new_column] = text[j]

    if i % save_interval == 0:
        data.to_csv(data_location, index = False)
    
data.to_csv(data_location, index = False)