from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

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
    - apply causal masking, crucial for causal self-attention, prevents tokens from attending future tokens
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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # set scale flag (kinda jank)
        # regularization
        self.n_head = config.n_head # number of heads
        self.n_embd = config.n_embd # dimensionality of token
        # more of a mask than 'bias', but following HF naming
        # bias is the "causal mask" used to prevent attention to future tokens
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
        
        # """ Initial Attention """

        # # scaled dot-product attention calculation
        # # k.transpose swaps last 2 dimensions -> (B, nh, hs, T)
        # # q @ new k performs batch matrix multiplication, contains raw attention scores
        # # att[b, h, i, j] = how much query q at i (for b and h) attends to key k at j
        # # (1.0 / math.sqrt(k.size(-1))) is scaling factor, k.size(-1) = hs
        # # prevents dot products from becoming too large, helps softmax gradients
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # autoregressive mask ensures tokens only draw from previous tokens
        # # masked_fill replaces all instances of 0 with -inf
        # # self.bias[: = take all, :T = take from 0 to (excluding) T], T = current sequence length
        # # -inf becomes 0 after softmax is applied
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # softmax function along last dimension (T)
        # # converts raw attention scores into probability distribution
        # # sum of all weights in query q = 1
        # att = F.softmax(att, dim=-1)

        # # result y = weighted sum of v vectors for each q, incorporating new context
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # """ FlashAttention """
        # OK SEEMS TO ONLY BE WORKING FOR NVIDIA (AGAIN)
        # run using: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python3 train_gpt2.py
        # turns attention into kernel fusion operation
        # supposedly 7.6x faster by respecting memory hierarchy
        # N x N matrix never gets materialized in HBM
        # Online rewrite of softmax algorithm allows for stored variables to speed up entire calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # set scale flag (kinda jank)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
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

    # keep your numbers nice, many powers of 2
    
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

        # weight sharing scheme
        # INPUT TOKENS AND OUTPUT HEAD SHARE WEIGHT
        # semantically similar tokens will have similar weights
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # if module is object of nn.Linear class
        if isinstance(module, nn.Linear):
            # DEFAULT mean = 0, standard deviation = 0.02, bias = 0
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2* because each layer has 2 blocks that add together (attn, mlp)
                # typical javier initialization sets std = 1/sqrt(incoming features into layer)
                # helps maintain reasonable scaling (~1) as model trains
                std *= (2 * self.config.n_layer) ** -0.5
            # 0.02 is roughly in vicinity, so hard coded here ok
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # bias not 0 by default
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # allows to generate text
    def forward(self, idx, targets=None):

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

        # calculate loss
        loss = None
        if targets is not None:
            # cross entropy loss function from pytorch
            # cross entropy cannot take multi dimensional tensors, so we flatten to 2d
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. any parameters that are 2d will be weight decayed, otherwise no
        # i.e. weight tensors in matmuls + embeddings decay, biases and layernorms no
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if available
        import inspect; fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    

import tiktoken

# loads data into chunks and iterates over document
# each epoch is iteration over entire document
# each chunk read is ineligible to be read again until next epoch
class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store in memory
        dataset = 'datasets/tiny_shakespeare.txt'
        with open(dataset, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # buffer of batch collection + 1
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:].view(B, T)) # targets
        # advance position in tensor
        self.current_position += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y


# -------------------------------------

import time

num_return_sequences = 5
max_length = 30


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# new gen apple cpu
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
# device = "cpu" # OVERRIDE

# get a data batch
# import tiktoken
# dataset = 'datasets/tiny_shakespeare.txt'
# enc = tiktoken.get_encoding('gpt2')
# with open(dataset, 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32 # (4 batches, 32 tokens per row)
# # take buffer of 
# buf = torch.tensor(tokens[:B*T + 1])
# # to device makes a copy
# buf = buf.to(device) # MUST SET buf = new instance of buf on new device
# # creates input buffer
# x = buf[:-1].view(B, T)
# # creates label buffer, offset from input by 1 index
# # displays the next token in each index
# y = buf[1:].view(B, T)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# initialize first training batch
train_loader = DataLoaderLite(B=8, T=512)

# seems to be nvidia architecture specific data type
# torch.set_float32_matmul_precision('high') # enable tensor 32

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
# good practice put model to evaluation mode when not training
# model.eval() # here it does nothing though
# move model to device
model.to(device)
# calculate logits and loss in nn.module forward pass
# pass in input and labels
# logits, loss = model(x, y)

# compiler for neural networks
# costs compilation time, but increases speed of calculations, reduces individual read/write overhead
# like a c compiler but for neural network, introduces kernel fusion
# prevents vram travel time (common bottleneck) by combining likewise operations in gpu cores
# read and write once for extensive calculations isntead of round trip for each operation
# i.e. (a * b) + c becomes 1 operation instead of 2
# chips in gpu cores have memory, but majority of gpu memory is in hbm (vram) 
model = torch.compile(model)

# aim not to favor any token too much at initialization
# if 50257 potential tokens, 1 should have probabilitiy of 1/50257
# thus, loss ~ -ln(1/50257) ~ 10.825
# must check before training

# learning rate has cosine decay from gpt-3 paper
max_lr = 6e-4
min_lr = max_lr * 0.1 # 10%
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3. in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 then goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer set up in pytorch
# lr=3e-4, eps=1e-8 default for debugging
# betas=(0.9, 0.95) also found in gpt-3 paper
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# backward pass
for step in range(max_steps):
    t0 = time.time() # start iteration timer

    # get next batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # reset gradients for backward pass
    
    # autocast with mixed precision for bf16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward() # deposits gradients, must equal zero
    # clips global norm of gradient to 1.0
    # prevents model from getting shocked by too large magnitudes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # set learning rate
    optimizer.step() # updates parameters and decrease loss

    # cpu default directs task to gpu and then moves on
    # sync with gpu before stopping clock
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert to ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    # loss is single value tensor living on device
    # calling .item ships 1D tensor back to cpu who converts into float 
    print(f"step {step:4d}, loss: {loss.item():.6f}, norm: {norm:.4f}, lr {lr:.4e}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

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
    print(">", decoded)