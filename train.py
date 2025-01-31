import torch

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

#Define a context length and batch size
context_length = 8
batch_size = 4

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
    print(inds)
    
get_batch("train")