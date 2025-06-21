from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CasualSelfAttention(nn.Module):

    """
    See "Let's build the GPT Tokenizer" video for context.
    Optimized version of same concepts.

    Implements "Masked Multi-Head Self-Attention" mechanism in decoder block
    
    - takes input x (sequence of token embeddings)
    - projects into query key, value
        - query (q) = what I'm looking for
        - key (k) = what I contain
        - value (v) = what information I carry
    - split qkv projections into n_head smaller "attention heads", focus on different aspects of x
    - calc attention scores, for each q compute how relevant it is to every other k in sequence
        - typically done via dot product
    - apply casual masking, crucial for casual (or causal) self-attention, prevents tokens from attending future tokens
        - here we use tril, future values = -inf
    - normalize scores with softmax function, turns relevance scores into probability distribution
        - indicates how much "attention" should be placed on each token
    - use attention probabilities to take weighted sum of v vectors, mixes info from relevant tokens into current
    - concatenate and project, combine all attention head outputs and project back into n_embd dimensionality
    - x = original input size n_embd, y = refined and contextualized input size n_embd

    in every attention block theres multiple heads running in parallel
    heads are parallel streams, outputs concatenated -> output of multi-head attention

    queries and keys interact multiplicatively to find out how important each one is 
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # sanity check, n_embd will be split equally among heads
        # key, query, value projections for all heads, but in a batch
        # projects input x of shape (B, T, n_embd) and projects into combined qkv tensor
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head # number of heads
        self.n_embd = config.n_embd # dimensionality of token
        # more of a mask than 'bias', but following HF naming
        # bias is the "casual mask" used to prevent attention to future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward
        # nh (number of heads n_head), hs (head size), C (number of channels) = nh * hs
        # GPT-2 (124M) has nh=12, hs=64, so C=nh*hs=768 channels in Transformer
        
        # queries and keys interact multiplicatively to find out how important each one is 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # splits along dim=2 into 3 equal parts 

        # view reshapes tensor to (B, T, nh, hs), transpose swaps dim[1] and dim[2]
        # treats nh as batch dimension to achieve head parallel streams when @
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all queries and keys)

        # scaled dot-product attention calculation
        # k.transpose swaps last 2 dimensions -> (B, nh, hs, T)
        # q @ new k performs batch matrix multiplication, contains raw attention scores
        # att[b, h, i, j] = how much query q at i (for b and h) attends to key k at j
        # (1.0 / math.sqrt(k.size(-1))) is scaling factor, k.size(-1) = hs
        # prevents dot products from becoming too large, helps softmax gradients
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # autoregressive mask ensures tokens only draw from previous tokens
        # masked_fill replaces all instances of 0 with -inf
        # self.bias[: = take all, :T = take from 0 to (excluding) T], T = current sequence length
        # -inf becomes 0 after softmax is applied
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # softmax function along last dimension (T)
        # converts raw attention scores into probability distribution
        # sum of all weights in query q = 1
        att = F.softmax(att, dim=-1)

        # result y = weighted sum of v vectors for each q, incorporating new context
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # transpose (B, nh, T, hs) -> (B, T, nh, hs)
        # contiguous = sharing border in memory, stored continuously
        # view(B, T, C) concatenates outputs from all nh attention heads along last dimension 
        # remember C (number of channels) = nh * hs
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # activation function, tanh approx uncommon but used for gpt 2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    modeled after hf transformer library
    """
    
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50k BPE merges + 256 hyte tokens + 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    
    # allows to generate text
    def forward(self, idx):

        # idx is shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is max sequence length."

        # forward token and position embeddings
        # creates range of indexes from 0 to T, same device as idx
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # included broadcasting since adding different shape

        # forward blocks to transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # last layer norm
        # with input (B, T), calculate next token (B, T+1)
        # vocab_size = number of possible tokens
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embd are determined from model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    


# -------------------------------------

num_return_sequences = 5
max_length = 30

# load ROCm through CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA (ROCm) not available, falling back to CPU.")

model = GPT.from_pretrained('gpt2')
# good practice put model to evaluation mode when not training
model.eval() # here it does nothing though
# move model to device
model.to(device)


# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language mode,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8, ), encoded is 8 tokens long
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8), 5 sequences * 8 tokens
# looks for the 9th token in each of 5 rows
x = tokens.to('cuda')

# generate, x = (B, T) = (5, 8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# loop adding new logits (colums of x) until max length is reached
while x.size(1) < max_length:
    # forward the model to get logits
    # no_grad tells pytorch to not prepare for backpropagation, saves memory and space
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size), wasteful but correct
        # get probabilities of next logit
        probs = F.softmax(logits, dim=-1)
        # top-k sampling of 50 (huggingface pipeline default)
        # essentially samples only top 50 probabilities, rest clamp to 0 and renormalize
        # avoids from going off the rails
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decode)

    # so we're not using cuda