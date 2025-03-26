import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import sys
sys.path.append("/work/10509/ptc487/vista/research/gpt2/code/modules")
from model import gpt2


#hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = True)
vocab_size = tokenizer.vocab_size

#These need to be the same as the checkpoint you are loading in
embed_dim = 768             #gpt2 124M had embed_dim = 768
context_length = 1024       #gpt2 124M had context_length = 1024
num_heads = 12              #gpt2 124M had num_heads = 12
activation = nn.GELU()      #gpt2 124M had activation = nn.GELU()
num_decoders = 12           #gpt2 124M had num_decoders = 12
dropout = .1

checkpoint_location = "../../checkpoints/subjects/checkpoint-1.pt"
prompts = ["A legal court case might be", "A common phrase in the english language is", "An example of a science discovery is", "Please tell me more about the various fields of mathematics and how they are inter related."]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the model
my_dict = torch.load(checkpoint_location)
model = gpt2(vocab_size, embed_dim, context_length, tokenizer, device, num_heads, activation, dropout, num_decoders)
model.load_state_dict(my_dict["model_state"])
model.to(device)

#Generate based on the prompts
for prompt in prompts:
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))).to(device)
    model_output = model.generate(input_ids)
    print(f"Prompt:\n{prompt}")
    print()
    print(f"Model Response:\n{model_output}")
    print()
    print()
    print()