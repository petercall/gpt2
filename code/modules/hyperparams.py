import torch

#Model hyperparameters
model_args = {               
    "embed_dim" : 768,              #gpt2 124M had embed_dim = 768
    "num_heads" : 12,               #gpt2 124M has num_heads = 12
    "num_decoders" : 12,            #gpt2 124M had num_decoders = 12
    "activation" : "gelu",          #gpt2 used "gelu"
    "dropout" : 0.1
}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Tokenizer hyperparameters
tokenizer_name = "gpt2"
tokenizer_args = {"clean_up_tokenization_spaces" : True}

#Dataset hyperparameters
data_location = "/work/10509/ptc487/vista/data/subjects"
file_names = {"train" : "train", "validation" : "validation"}     #Do NOT include .txt at the end, just the name of the file. Adjust only the values, NOT the keys
batch_size = 24
token_window_slide = 512     #I set this to be context_length/2 (so 1024/2=512) so that it slides halfway over the context window each time
