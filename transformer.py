import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


#Hyperparameters
batch_size = 64
context_length = 128
embed_dim = 256
num_heads = 2
dropout = 0
activation = "relu"
causal_masking = True
padding_mask = torch.randint(low=0, high = 2, size = (batch_size, context_length))
padding_mask = None
input_tens = torch.randn(batch_size, context_length, embed_dim)
#----------------------


#Define the model
model = torch.nn.TransformerEncoderLayer(embed_dim,num_heads, dropout = dropout, activation = activation, batch_first = True, norm_first = True)

# Get the parameters of the model
params = [(name, params) for name, params in model.named_parameters()]

#Transformer output
if causal_masking:
    if padding_mask is not None:
        transformer_output = model(input_tens, src_mask = nn.Transformer.generate_square_subsequent_mask(context_length).to(torch.bool), is_causal = True, src_key_padding_mask = padding_mask.to(torch.bool))
    else:
        transformer_output = model(input_tens, src_mask = nn.Transformer.generate_square_subsequent_mask(context_length).to(torch.bool), is_causal = True)
else:
    if padding_mask is not None:
        transformer_output = model(input_tens, src_key_padding_mask = padding_mask.to(torch.bool))
    else:
        transformer_output = model(input_tens)

#Layer Norms
layer_norm1 = nn.LayerNorm([embed_dim])
layer_norm2 = nn.LayerNorm([embed_dim])
layer_norm1.weight = params[8][1]
layer_norm1.bias = params[9][1]
layer_norm2.weight = params[10][1]
layer_norm2.bias = params[11][1]

#Linear Layers
linear1 = nn.Linear(embed_dim, 2048)
linear2 = nn.Linear(2048, embed_dim)
linear1.weight = params[4][1]
linear1.bias = params[5][1]
linear2.weight = params[6][1]
linear2.bias = params[7][1]

#in_projection and out_projection layer for the self attention layer
in_proj = nn.Linear(embed_dim, 3*embed_dim)
in_proj.weight = params[0][1]
in_proj.bias = params[1][1]
out_proj = nn.Linear(embed_dim, embed_dim)
out_proj.weight = params[2][1]
out_proj.bias = params[3][1]

#self attention block
x = layer_norm1(input_tens)
x = in_proj(x)
Q = x[:,:,:embed_dim].view(batch_size, context_length, num_heads, embed_dim//num_heads).transpose(1,2)
K = x[:,:,embed_dim:2*embed_dim].view(batch_size, context_length, num_heads, embed_dim//num_heads).transpose(1,2)
V = x[:,:,2*embed_dim:3*embed_dim].view(batch_size, context_length, num_heads, embed_dim//num_heads).transpose(1,2)
attn = (Q @ K.transpose(2,3))/torch.sqrt(torch.tensor(embed_dim//num_heads, dtype=torch.float32))

if causal_masking:
    causal_mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
    attn = attn.masked_fill(causal_mask == 1, float("-inf"))

if padding_mask is not None:
    attn = attn.masked_fill(padding_mask[:, None, None, :] == 1, float("-inf"))

x = F.softmax(attn, dim = -1) 
x = x @ V
x = x.transpose(1,2).reshape(batch_size, context_length, embed_dim)
x = out_proj(x)
x += input_tens

#feed forward block
y = layer_norm2(x)
y = F.relu(linear1(y))
y = linear2(y)
manual_output = x+y

# Test for similarity
print(torch.allclose(transformer_output, manual_output, atol=1e-06, equal_nan=True))