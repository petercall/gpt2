import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AutoTokenizer
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modules")))
from dataset import TextDataset
from model import gpt2
from training import train, validation_func, graph_losses

#hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
logging.getLogger("transformers").setLevel(logging.ERROR)   #Used to supress the error the tokenizer gives when we tokenize the entire training sequence all at once.
tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = True)
vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

data_location = "../../data/subjects/intermediate_text"
file_names = ["intermediate_train.txt", "intermediate_val.txt"]

batch_size = 32              #Not sure what gpt2 used
embed_dim = 768              #gpt2 124M had embed_dim = 768
context_length = 1024        #gpt2 124M has context_length = 1024
token_window_slide = context_length // 2 #This is how many tokens each input data point slides over
    
num_heads = 12               #gpt2 124M had num_heads = 12
num_decoders = 12            #gpt2 124M had num_decoders = 12
activation = "gelu"          #gpt2 used "gelu"
dropout = 0.1

epochs = 40
val_interval = 1
smallest_val_loss = float("inf")
stop_patience = 7          #How many validation loops with no decrease in validation loss before the training stops

scheduler_patience = 4      #How many validation loops with no decrease in validation loss before the learning rate is multiplied by factor
scheduler_factor = .3        #The factor that the learning rate gets multiplied by when it is not improving

checkpoint_storage_location = "../../checkpoints/subjects/checkpoint-intermediate-2.pt"
graph_save_location = "../../outputs/graphs"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Input Code-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#load in the train, validation, and test text
data = dict()
for current_file in file_names:
    with open(os.path.join(data_location, current_file), "r") as file:
        data[current_file[:current_file.find(".")]] = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(file.read())))


#Create the train and val dataset and the train and val dataloader
train_data = TextDataset(data["intermediate_train"], context_length, token_window_slide)
val_data = TextDataset(data["intermediate_val"], context_length, token_window_slide)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, pin_memory = True)

#Create the model and optimizer
gpt = gpt2(vocab_size, embed_dim, context_length, tokenizer, device, num_heads, activation, dropout, num_decoders).to(device)
gpt.to(device)

#Create the optimizer, scheduler, and loss function
optimizer = optim.AdamW(gpt.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = scheduler_patience, factor = scheduler_factor)
loss_function = nn.CrossEntropyLoss()

# Train the model
train_losses, val_losses, smallest_val_loss = train(
    gpt,                #model
    optimizer,          #optimizer
    scheduler,          #lr scheduler
    loss_function,      #loss function
    device,             #torch device
    train_loader,       #train loader
    val_loader,         #val loader
    validation_func,    #function used for validation
    epochs,             #epochs to train for
    val_interval,       #How often to validate
    smallest_val_loss,  #current smallest validation loss
    stop_patience,      #Number of validation runs with no improvement after which we terminate the training loop
    checkpoint_storage_location,    #location to store checkpoint
    )

print(f"Smallest Val loss is: {smallest_val_loss}")

# #Save graphs of the losses
graph_losses(train_losses, "Train", graph_save_location)
graph_losses(val_losses, "Validation", graph_save_location)