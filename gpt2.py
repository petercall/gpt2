import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


#hyperparameters
small = False
print_vals = False

if small:
    batch_size = 2
    context_length = 4
    embed_dim = 8
else:
    batch_size = 128
    context_length = 64
    embed_dim = 256
    
num_heads = 8
num_decoders = 10
activation = "gelu"
dropout = 0.1

epochs = 10000
eval_interval = 1000
eval_iterations = 400

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1337)
#---------------

#region
if print_vals:
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    print(f"Embedding dimension: {embed_dim}")
    print()

#load in the train, validation, and test text
with open("data/train.txt", "r") as file:
    train_text = file.read()
with open("data/validation.txt", "r") as file:
    validation_text = file.read()
with open("data/test.txt", "r") as file:
    test_text = file.read()
    
#Create the list of characters and the vocabulary size
chars = sorted(list(set(train_text)))
vocab_size = len(chars)

#Create two dictionaries and the encoder and decoder
char_to_index = dict(zip(chars, range(vocab_size)))
index_to_char = dict(zip(range(vocab_size), chars))
encode = lambda x: [char_to_index[char] for char in x]
decode = lambda x: "".join([index_to_char[char] for char in x])

#Tokenize the training text
train_data = torch.tensor(encode(train_text), dtype = torch.long)
val_data = torch.tensor(encode(validation_text), dtype = torch.long)
test_data = torch.tensor(encode(test_text), dtype = torch.long)

#Define a function to get a batch of inputs
def get_batch(split: str):
    if split == "train":
        data = train_data
    elif split == "validation":
        data = val_data
    elif split == "test":
        data = test_data
    else:
        raise ValueError("Input must be one of the following: 'train', 'validation', 'test")
    
    inds = torch.randint(high = len(data) - context_length, size = (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in inds])
    y = torch.stack([data[i+1:i+context_length+1] for i in inds])
    
    return x, y
#endregion


class gpt2(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(gpt2, self).__init__()
        #Define the embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(vocab_size, embed_dim)
        
        #Define the transformer
        self.encoding_layer = nn.TransformerEncoderLayer(embed_dim, nhead = num_heads, activation = activation, dropout = dropout, batch_first = True, norm_first = True)
        self.transformer = nn.TransformerEncoder(self.encoding_layer, num_decoders, enable_nested_tensor = False)
        
        #Define the prediciton head
        self.pred_head = nn.Linear(embed_dim, vocab_size)
        
        
    def forward(self, idx, padding_mask=None):
        #idx = (batch_size, context_length)
        
        #Get the token emebedding: self.embedding(idx) is (batch_size, context_length, embed_dim)
        tok_emb = self.tok_emb(idx)
        
        #Get the positional embedding
        pos_emb = self.pos_emb(torch.arange(idx.shape[1]).to(device))
        
        #Add the token and positional embedding
        x = tok_emb + pos_emb
        
        #Pass it through the transformer layer
        if padding_mask is not None:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(context_length).to(device),
                is_causal = True,
                src_key_padding_mask = padding_mask
            )
        else:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(context_length).to(device),
                is_causal = True
            )
        
        #Pass it through the prediction head: x[:,-1,:] is of shape (batch_size, embed_dim).
        #Passing it tourhg the prediction head causes output to be of shape: (batch_size, vocab_size)
        logits = self.pred_head(x[:,-1,:])
        
        return logits
    
    def generate(self, input_text: str, max_new_tokens = 200):
        with torch.no_grad():
            
            idx = torch.tensor(encode(input_text), dtype = torch.long).unsqueeze(0).to(device)
                      
            for _ in range(max_new_tokens):
                #Push the sequence through the model and get the probabilities
                logits = self(idx).squeeze()
                probs = F.softmax(logits, dim = 0)
                
                #Sample from the probability distribution to get the next token
                idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(1)
                
                #Place idx_next at the end of idx
                idx = torch.cat((idx, idx_next), dim = 1)
            
            #Turn the tokens back into a string
            output = decode(idx.squeeze().detach().cpu().numpy())

        return output
    
    
#Define an evaluation function
@torch.no_grad()
def eval(model, iterations):
    model.eval()
    eval_losses = []
    for _ in range(iterations):
        #Get the data
        x_batch, y_batch = get_batch("validation")
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        #For now I am not doing padding, so just grab the last tokens in y_batch
        y_batch = y_batch[:,-1]
        
        #Push x_batch through the model and calculate the loss
        logits = model(x_batch)
        eval_loss = F.cross_entropy(logits, y_batch)
        
        eval_losses.append(eval_loss.item())
        
    model.train()    
        
    return np.average(eval_losses)
    
        
#Define a training loop function
def train(model, optimizer, epochs):
    #Define the loss variables
    lowest_training_loss = np.inf
    lowest_eval_loss = np.inf
    training_losses = []
    eval_losses = []
    
    model.train()
    for i in range(epochs):
        #Grab a batch of data
        x_batch, y_batch = get_batch("train")
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)    
        
        #For now I am not doing padding, so just grab the last tokens in y_batch
        y_batch = y_batch[:,-1]

        #Zero out the gradients
        optimizer.zero_grad()

        #Push the batch through the model and calculate the loss
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        
        #Take the step
        loss.backward()
        optimizer.step()
        
        #Add the loss to the list, and save a new lowest loss if necessary
        training_losses.append(loss.item())
        if loss.item() < lowest_training_loss:
            lowest_training_loss = loss.item()
            
        if i % eval_interval == 0:
            eval_loss = eval(model, eval_iterations)
            print(f"Step: {i}, Train Loss: {round(loss.item(),3)}, Eval Loss: {round(eval_loss.item(), 3)}")
            
            eval_losses.append(eval_loss.item())
            if eval_loss.item() < lowest_eval_loss:
                lowest_eval_loss = eval_loss.item()
                
    print(f"Iteration: {i}, Lowest Training Loss: {round(lowest_training_loss,3)}, Lowest Test Loss: {round(lowest_eval_loss,3)}")
    print()
                
    return training_losses, eval_losses

#Define a function to graph the losses
def graph_losses(losses: list, name: str):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel(f"{name} Loss")
    plt.title(f"{name} Loss over Time")  
    plt.savefig(f"{name}_Loss_{len(losses)*(epochs//eval_interval)}_Epochs.png")  
    plt.clf()


#Create the model and optimizer
gpt = gpt2(vocab_size, embed_dim).to(device)
optimizer = optim.AdamW(gpt.parameters())

# Train the model
training_losses, eval_losses = train(gpt, optimizer, epochs)

#Save graphs of the losses
graph_losses(training_losses, "Training")
graph_losses(eval_losses, "Testing")

# #Generate and print out a text sample
new_text = gpt.generate("How can this be done?", max_new_tokens = 30)
print()
print(new_text)
print()