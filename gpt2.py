import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import os
from tqdm.auto import tqdm
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)   #Used to supress the error the tokenizer gives when we tokenize the entire training sequence all at once.


#hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = True)
vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

data_location = "data/shakespeare"
file_names = ["train.txt", "validation.txt", "test.txt"]

batch_size = 2            #Not sure what gpt2 used
context_length = 64       #gpt2 124M has context_length = 1024
embed_dim = 128             #gpt2 124M had embed_dim = 768
    
num_heads = 4              #gpt2 124M had num_heads = 12
num_decoders = 2           #gpt2 124M had num_decoders = 12
activation = "gelu"         #gpt2 used "gelu"
dropout = 0.1

epochs = 2
val_interval = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#Functions/Class Definitions------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
class gpt2(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, tokenizer):
        super(gpt2, self).__init__()
        #Save the tokenizer to be used in the generate function
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        #Define the embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        
        #Define the transformer
        self.transformer_layer = nn.TransformerEncoderLayer(embed_dim, nhead = num_heads, activation = activation, dropout = dropout, batch_first = True, norm_first = True, dim_feedforward = 4 * embed_dim)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_decoders, enable_nested_tensor = False)
        self.layerNorm = nn.LayerNorm(embed_dim)
        
        #Define the prediciton head
        self.pred_head = nn.Linear(embed_dim, vocab_size)
        
        
    def forward(self, idx, padding_mask=None):
        #idx is of shape = (B, T)
        if idx.shape[1] > self.context_length:
            raise ValueError(f"Cannot input text greater than {self.context_length} number of tokens.")
        
        tok_emb = self.tok_emb(idx)  #Get the token emebedding: self.embedding(idx) is (B, T, embed_dim)
        pos_emb = self.pos_emb(torch.arange(idx.shape[1]).to(device)) #Get the positional embedding which is of shape: (T, embed_dim)
        x = tok_emb + pos_emb
        
        #Pass it through the transformer layer
        if padding_mask is not None:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(torch.bool).to(device),
                is_causal = True,
                src_key_padding_mask = padding_mask.to(torch.bool)
            )
        else:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device),
                is_causal = True
            )
        #The output is still (B, T, embed_dim)
        
        #We then put it through a linear head to get the logits, which converts it to shape (B, T, vocab_size)
        logits = self.pred_head(self.layerNorm(x))
            #Output size: (B, T, vocab_size)
        
        return logits
    
    def generate(self, input_text: str, max_new_tokens = 200, topk = 50):
        if not isinstance(input_text, str):
            raise TypeError("The input text needs to be of type string.")
            #This means that this funciton only accepts a single string as of now.
            #I can edit it later to accept a list of strings if I want to.
        
        with torch.no_grad():
            self.eval()
            
            #Tokenize the input text
            output_dict = self.tokenizer(input_text)
            token_ids = torch.tensor(output_dict["input_ids"]).to(device).unsqueeze(0)   #token_ids is of shape (1, num_tokens_in_input_text)

            #Throw an error if the input text is longer than the maximum allowable tokens
            if token_ids.shape[1] > self.context_length:
                raise ValueError(f"You have entered text that is too long. The maximum context length is: {self.context_length}")

            next_token_id = -1

            for _ in range(max_new_tokens):
                if next_token_id == self.tokenizer.eos_token:
                    break
                
                #Push the sequence through the model and get the probabilities
                logits = self(token_ids).squeeze()      #logits is of shape (num_tokens_in_input_text, vocab_size)  
                logits = logits[-1,:].squeeze()         #We only care about the logits associated with the LAST token embedding. Logits now has shape (vocab_size)  
                probs = F.softmax(logits, dim = -1)     #probs is of shape (vocab_size)
                k_probs, k_indices = torch.topk(probs, topk)  #k_probs and indices are of shape (topk)
                
                #Sample from the probability distribution to get the next token
                index_k = torch.multinomial(k_probs, num_samples=1)
                
                #Get the token index
                next_token_id = k_indices[index_k]
                
                #Place idx_next at the end of idx
                token_ids = torch.cat((token_ids, next_token_id.unsqueeze(1)), dim = 1)
            
            #Turn the token_ids back into a string
            output_text = self.tokenizer.decode(token_ids.squeeze(), clean_up_tokenization_spaces=True)

        return output_text
    
    
#Define a validation function
@torch.no_grad()
def validation_func(model, loss_func, device, val_loader):
   model.eval()
   val_loop_losses = []
  
   for x, y in val_loader:
       x, y = x.to(device), y.to(device)
      
       logits = model(x)    #logits has shape (B, T, vocab_size)
       val_loop_losses.append(loss_func(logits.view(-1, logits.shape[-1]), y.view(-1)).item())    #We reshape logits to be of shape (B*T, vocab_size) and targets to be of shape (B*T)
      
   return torch.mean(val_loop_losses)
    
        
#Define a training loop function
def train(model, optimizer, scheduler, loss_func, device, train_loader, val_loader, validation_func, epochs, val_interval, smallest_val_loss, patience, checkpoint_filepath, print_losses):
    train_losses = []
    val_losses = []
    patience_count = 0
    
    for epoch in tqdm(range(epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)      
            optimizer.zero_grad()

            logits = model(x)   #of shape (B, T, vocab_size)
            loss = loss_func(logits.view(-1, logits.shape[-1]), y.view(-1))   #Reshape to size (B*T, vocab_size) and (vocab_size), which is what CELoss is expecting
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()


        if epoch % val_interval == 0:
            val_loss = validation_func(model, loss_func, device, val_loader)
            val_loss.append(val_loss.item())


            if val_loss.item() < smallest_val_loss:
                patience_count = 0     
                smallest_val_loss = val_loss.item()
                checkpoint_dict = {
                    "epoch" : epoch,
                    "model_state" : model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict()
                }
                torch.save(checkpoint_dict, checkpoint_filepath)
            else:
                patience_count += 1
                if patience_count > patience:
                    return train_losses, val_losses, smallest_val_loss, epoch+1
                
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            if print_losses:
                print(f"Training Loss: {round(loss.item(),3)}, Validation Loss: {round(val_loss.item(),3)}")
                print()
      
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    return train_losses, val_losses, smallest_val_loss, epochs     


#Define a function to graph the losses
def graph_losses(losses: list, name: str, epochs: int):  #name should be "train" or "validation" or "test"
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel(f"{name} Loss")
    plt.title(f"{name} Loss over Time")  
    if name == "train" or name == "test":
        plt.savefig(f"{name}_loss_{len(losses)}_Epochs.png")  
    elif name == "validation":
        plt.savefig(f"{name}_loss_{len(losses)*(epochs//val_interval)}_Epochs.png")  
    plt.clf()
    
class TextDataset(Dataset):
    def __init__(self, text, context_length):
        super(TextDataset, self).__init__()
        self.data = text
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, i):
        return self.data[i : i+self.context_length], self.data[i+1 : i+self.context_length+1]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#Input Code-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#load in the train, validation, and test text
data = dict()
for current_file in file_names:
    with open(os.path.join(data_location, current_file), "r") as file:
        data[current_file[:current_file.find(".")]] = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(file.read())))


#Create the train and val dataset and the train and val dataloader
train_data = TextDataset(data["train"], context_length)
val_data = TextDataset([data["validation"]], context_length)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, pin_memory = True)

#Create the model and optimizer
gpt = gpt2(vocab_size, embed_dim, context_length, tokenizer).to(device)
gpt.to(device)

#Create the optimizer, scheduler, and loss function
optimizer = optim.AdamW(gpt.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size = 30)
loss_function = nn.CrossEntropyLoss()

# Train the model
training_losses, val_losses, smallest_val_loss, trained_epochs = train(
    gpt, 
    optimizer, 
    scheduler, 
    loss_function, 
    device, 
    train_loader, 
    val_loader, 
    validation_func, 
    epochs, 
    val_interval, 
    float("inf"), 
    10, 
    "checkpoints/checkpoint.pt",
    True
    )


# #Save graphs of the losses
graph_losses(training_losses, "train", trained_epochs)
graph_losses(val_losses, "validation", trained_epochs)

# #Generate and print out a text sample
new_text = gpt.generate("Where goest thou?", max_new_tokens = 30)
print()
print(new_text)
print()