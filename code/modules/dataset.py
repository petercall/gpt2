#Regular imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)   #Used to supress the error the tokenizer gives when we tokenize the entire training sequence all at once.



#File imports
from .hyperparams import data_location, file_names, token_window_slide, batch_size
from .model import tokenizer


class TokenIDDataset(Dataset):
    def __init__(self, data, context_length, slide_length):
        super(TokenIDDataset, self).__init__()
        self.data = data
        self.context_length = context_length
        self.slide_length = slide_length

    def __len__(self):
        return (self.data.shape[0]//self.slide_length) - 1

    def __getitem__(self, i):
        start = i*self.slide_length
        return self.data[start : start+self.context_length], self.data[start+1 : start+self.context_length+1]
    
    
# load in the train, validation, and test text
data = dict()
for current_file in file_names.keys():
    with open(os.path.join(data_location, file_names[current_file] + ".txt"), "r") as file:
        data[current_file] = tokenizer(file.read(), return_tensors = "pt", truncation = False, padding = False)

#Create the train and val dataset
train_data = TokenIDDataset(data["train"]["input_ids"][0], tokenizer.model_max_length, token_window_slide)
val_data = TokenIDDataset(data["validation"]["input_ids"][0], tokenizer.model_max_length, token_window_slide)

#Input the train and validation data into the DataLoader
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
val_loader = DataLoader(val_data, batch_size = batch_size, pin_memory = True, num_workers = 4)