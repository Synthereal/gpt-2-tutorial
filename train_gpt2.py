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
class GPTconfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

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