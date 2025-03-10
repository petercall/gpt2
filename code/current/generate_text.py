import torch
import torch.nn as nn
from transformers import AutoTokenizer
sys.path.append("/work/10509/ptc487/vista/research/gpt2/code/modules")
from model import gpt2

#hyperparameters---------------------------------------------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = True)
vocab_size = tokenizer.vocab_size

embed_dim = 768             #gpt2 124M had embed_dim = 768
context_length = 1024       #gpt2 124M had context_length = 1024
num_heads = 12              #gpt2 124M had num_heads = 12
activation = nn.GELU()      #gpt2 124M had activation = nn.GELU()
num_decoders = 12           #gpt2 124M had num_decoders = 12
dropout = .1

checkpoint_location = "../../checkpoints/subjects/checkpoint-intermediate-1.pt"
prompt = "A common phrase in the english language is"
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



my_dict = torch.load(checkpoint_location)
model = gpt2(vocab_size, embed_dim, context_length, tokenizer, device, num_heads, activation, dropout, num_decoders)
model.load_state_dict(my_dict["model_state"])
model.to(device)

input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))).to(device)
model_output = model.generate(input_ids)
print(model_output)