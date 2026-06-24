# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt

# Setting hyperparameters
# batch_size = 64        # How many independent sequences to be processed in parallel
# block_size = 256       # maximum context length for prediction
# epochs = 5000
# eval_interval = 500
# learning_rate = 1e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embed = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

# --- Faster Hyperparameters ---
batch_size = 32        # REDUCED: 64 is huge for this. 32 gives faster step times.
block_size = 128       # REDUCED: Attention is quadratic (O(N^2)). 128 is 4x faster than 256!
epochs = 5000          # (Usually means 'max_iters' in Karpathy's code)
eval_interval = 500
learning_rate = 3e-4   # INCREASED: Smaller models can handle larger learning rates.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50        # REDUCED: You don't need 200 batches to estimate loss. 50 is plenty and saves massive time during eval spikes.

# --- Slimmer Architecture ---
n_embed = 256          # REDUCED: 384 is overkill for 1M words. 256 means much faster matrix math.
n_head = 4             # REDUCED: Must divide n_embed evenly (256 / 4 = 64 head size).
n_layer = 4            # REDUCED: 4 layers is plenty to learn basic grammar and structure.
dropout = 0.2

print(device)
# exit()

torch.manual_seed(13372)

f = open('shakespere.txt', 'r', encoding='utf-8')
text = f.read()

# Unique characters in the texts
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating encoding mappings (token embeddings)
stoi = {s:i for i, s in enumerate(chars)}
# itos = {i:s for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Traning and testing splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch creation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0,(len(data) - block_size), (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Multi head attention module
class MultiHeadAttention(nn.Module):

    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# Head of an attention module
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (T**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # Decoder block
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)
        out = wei @ v

        return out

# Feed Forward (simple multi layer perceptron)
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.net(x)

# Communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(head_size, n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Attention Model converted from Biagram Model from the Wavenet notebook
class AttentionLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Reading the logits off of the next token from the lookup table and then traning on that model
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_heads = MultiHeadAttention(4, n_embed//4)
        # self.ffwd = FeedForward(n_embed)
        # Instead of one we have multiple blocks of attention and feed forward blocks as per the diagram in the paper
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     nn.LayerNorm(n_embed),
        # )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx --> B
        # targets --> T
        # logits --> combination of BTC
        token_embeddings = self.token_embedding_table(idx)  # B T C
        pos_embeddings = self.position_embedding_table(torch.arange(T, device = device))
        
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)             # B T vocab_size

        # Reshaping logits
        if targets == None:
            loss = None                     # For generating
        else:
            B, T, C = logits.shape          # For training
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is the current context in a batch
        # makes B X T+1, B X T+2, B X T+3 and so on..
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]                           # (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)   
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        
        return idx
    
model = AttentionLanguageModel()
m = model.to(device=device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Optimiser (AdamW) which edits the learning rate as per the epochs
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Forward pass with backpropogation
for iter in range(epochs):

    # Sedom check the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']: .4f}, val loss {losses['eval']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# Sampling from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
f.close()