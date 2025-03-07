import torch
import torch.nn as nn
import torch.nn.functional as F

class gpt2(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, tokenizer, device, num_heads, activation, dropout, num_decoders):
        super(gpt2, self).__init__()
        #Save the tokenizer to be used in the generate function
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device
        
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
        pos_emb = self.pos_emb(torch.arange(idx.shape[1]).to(self.device)) #Get the positional embedding which is of shape: (T, embed_dim)
        x = tok_emb + pos_emb
        
        #Pass it through the transformer layer
        if padding_mask is not None:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(torch.bool).to(self.device),
                is_causal = True,
                src_key_padding_mask = padding_mask.to(torch.bool)
            )
        else:
            x = self.transformer(
                x, 
                mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(self.device),
                is_causal = True
            )
        #The output is still (B, T, embed_dim)
        
        #We then put it through a linear head to get the logits, which converts it to shape (B, T, vocab_size)
        logits = self.pred_head(self.layerNorm(x))
            #Output size: (B, T, vocab_size)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens = 200, topk = 50):
        #Reshape to (1, length) if it is of shape (length)
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
        
        #Throw an error if the input text is longer than the maximum allowable tokens
        if input_ids.shape[1] > self.context_length:
            raise ValueError(f"You have entered text that is too long. The maximum context length is: {self.context_length}")

        with torch.no_grad():
            self.eval()
            
            next_token_id = -1
            
            for _ in range(max_new_tokens):
                if next_token_id == self.tokenizer.eos_token:
                    break
                
                #Push the sequence through the model and get the probabilities
                logits = self(input_ids).squeeze()      #logits is of shape (num_tokens_in_input_text, vocab_size)  
                logits = logits[-1,:].squeeze()         #We only care about the logits associated with the LAST token embedding. Logits now has shape (vocab_size)  
                probs = F.softmax(logits, dim = -1)     #probs is of shape (vocab_size)
                k_probs, k_indices = torch.topk(probs, topk)  #k_probs and indices are of shape (topk)
                
                #Sample from the probability distribution to get the next token
                index_k = torch.multinomial(k_probs, num_samples=1)
                
                #Get the token index
                next_token_id = k_indices[index_k]
                
                #Place idx_next at the end of idx
                input_ids = torch.cat((input_ids, next_token_id.unsqueeze(1)), dim = 1)
            
            #Turn the token_ids back into a string
            output_text = self.tokenizer.decode(input_ids.squeeze().tolist(), clean_up_tokenization_spaces=True)
        
        return output_text