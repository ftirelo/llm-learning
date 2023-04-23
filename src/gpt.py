import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------
# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)


# --------------------------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


# --------------------------------
# Vocabulary, encoding, decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --------------------------------
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# --------------------------------
# Data loading
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


# Averages out the loss of the last `eval_iters` steps.
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


# --------------------------------
# Head module
class Head(nn.Module):
  '''One-head of self-attention.'''

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    # Not a parameter of the model, should be defined as a buffer.
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)  # (B, T, C)
    q = self.query(x)  # (B, T, 16)
    # Divide by sqrt(C) to avoid sharpening the weights in the direction
    # of the largest weight.
    wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, C) @ (B, C, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out


class MultiHeadAttention(nn.Module):
  '''Multi-head of self-attention in parallel.'''

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate over the channel dim
    out = self.proj(out)
    out = self.dropout(out)
    return out


class FeedForward(nn.Module):
  '''A simple linear layer followed by a non-linearity.'''

  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.ReLU(),
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  '''Transformer block: communication followed by computation.'''

  def __init__(self, n_embed, n_head):
    # n_embed: embedding dimension, n_head: number of heads we'd like
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))  # Apply one head self-attention. (B, T, C)
    x = x + self.ffwd(self.ln2(x))  # Apply linear layer followed by non-linearity
    return x


# --------------------------------
# First bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed)  # Final layer norm
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx)  # (B:batch, T:time, C:channel)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
    x = tok_emb + pos_emb  # B, T, C - with broadcasting
    x = self.blocks(x)  # (B, T, C)
    x = self.ln_f(x)  # (B, T, C)
    logits = self.lm_head(x)  # B, T, vocab_size
    
    if targets is None:
      loss = None
    else:
      # B is the mini-batch size
      # T corresponds to the number of characters in the block
      # C corresponds to the prediction (channel)
      # This is a bigram model, so we take the B*T sequences
      # as examples to the model. In the current configuration,
      # we have 4 blocks of 8 characters per mini-batch; each
      # block corresponds to X = (C_i, C_{i+1}, ..., C_{i+7})
      # and Y = X = (C_{i+1}, ..., C_{i+7}, C_{i+8}). Out of
      # those, take each (C_j, C_{j+1}) as a bigram in logits
      # and targets.
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # Ensure position embedding lookup doesn't run out of bounds.
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      # The next line makes the model focus on the last character
      # in the sequence only
      logits = logits[:, -1, :]  # Becomes (B, C)
      probs = F.softmax(logits, dim=-1)  # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx


model = BigramLanguageModel()
m = model.to(device)


# --------------------------------
# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
  # every once in a while evaluate the loss on train and val sets
  if iter % eval_iters == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    print(f'step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


# --------------------------------
# Generate from model

# Zero stands for the newline character, and that's how we
# kick-off generation; the first 1 corresponds to the batch
# size, so we are only generating one.
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
