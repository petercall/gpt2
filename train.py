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
context_length = 8
batch_size = 32
epochs = 40000
eval_interval = 10000
eval_iterations = 2500
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1337)
#---------------

#region
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

#Define the bigram model: This is simply turning the embedding table into a probability distribution for the next most likely word, according to the token number
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx):
        #idx comes in as size (batch_size, context_length sized vector). We are going to treat each element in idx as its own sample for the bigram model though, so reshape it to a 1D tensor.
        idx = idx.view(-1)
        
        #logits will be of shape (batch_size*context_length, vocab_size).
        logits = self.embedding(idx)
        
        return logits
    
    def generate(self, max_new_tokens = 200, starting_token = "\n", print = False):
        text = starting_token
        ind = torch.tensor(encode(starting_token)).to(device)
        ind_list = list(range(vocab_size))

        with torch.no_grad():
            for _ in range(max_new_tokens):
                #Push the index through the model, and get the next index by sampling from the characters according to the probability distribution that the model outputs
                logits = bigram_model(ind).squeeze()
                ind = torch.tensor(random.choices(ind_list, weights=F.softmax(logits, dim = 0))[0]).to(device)
                
                #Get the character associated with the index it spit out, and add it to the string
                new_char = decode([ind.item()])
                text += new_char
        
        if print:
            print(text)
            print()
        
        return text
     
#Define an evaluation function
@torch.no_grad()
def eval(model, iterations):
    model.eval()
    eval_losses = []
    for _ in range(iterations):
        #Get the data
        x_batch, y_batch = get_batch("validation")
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        #Push x_batch through the model and calculate the loss
        logits = model(y_batch)
        eval_loss = F.cross_entropy(logits, y_batch.view(-1))
        
        eval_losses.append(eval_loss.item())
        
    model.train()    
        
    return np.average(eval_losses)
        
#Define a function to graph the losses
def graph_losses(losses: list, name: str):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel(f"{name} Loss")
    plt.title(f"{name} Loss over Time")  
    plt.savefig(f"{name}_Loss_{len(losses)}Epochs.png")  
    plt.clf()
        
#Define a training loop function
def train(model, optimizer, epochs):
    #Define the loss variables
    lowest_training_loss = np.inf
    lowest_eval_loss = np.inf
    training_losses = []
    eval_losses = []
    
    model.train()
    for i in tqdm(range(epochs)):
        #Grab a batch of data
        x_batch, y_batch = get_batch("train")
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)    

        #Zero out the gradients
        optimizer.zero_grad()

        #Push the batch through the model and calculate the loss
        logits = bigram_model(x_batch)
        loss = F.cross_entropy(logits, y_batch.view(-1))
        
        #Take the step
        loss.backward()
        optimizer.step()
        
        #Add the loss to the list, and save a new lowest loss if necessary
        training_losses.append(loss.item())
        if loss.item() < lowest_training_loss:
            lowest_training_loss = loss.item()
            
        if i % eval_interval == 0:
            eval_loss = eval(bigram_model, eval_iterations)
            print(f"Step: {i}, Train Loss: {round(loss.item(),3)}, Eval Loss: {round(eval_loss.item(), 3)}")
            
            eval_losses.append(eval_loss.item())
            if eval_loss.item() < lowest_eval_loss:
                lowest_eval_loss = eval_loss.item()
                
    print(f"Lowest Training Loss: {lowest_training_loss}, Lowest Test Loss: {lowest_eval_loss}")
                
    return training_losses, eval_losses



#Create the model and optimizer
bigram_model = BigramModel(vocab_size).to(device)
optimizer = optim.AdamW(bigram_model.parameters())

#Train the model
training_losses, eval_losses = train(bigram_model, optimizer, epochs)
training_losses, eval_losses = train(bigram_model, optimizer, 100)

#Save graphs of the losses
graph_losses(training_losses, "Training")
graph_losses(eval_losses, "Testing")

#Generate and print out a text sample
new_text = bigram_model.generate(print = True)