import torch
import torch.nn as nn
from torch.nn import functional as F
import csv
import time
import os
import random
from itertools import combinations

# hyperparameters
batch_size = 64
block_size = 32
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# --- NEW HYPERPARAMETERS FOR ORTHOGONALITY ---
ortho_lambda = 0.1  # How strongly to enforce orthogonality
ortho_margin = 0.0  # Soft margin (0.2 means angles between 78 and 102 degrees are unpenalized)
output_dir = "outputs_ortho_wv" # Save to a new folder to compare with vanilla
# ---------------------------------------------

torch.manual_seed(1337)

train_metrics_rows = []
eval_metrics_rows = []
head_activation_rows_by_block = {i: [] for i in range(n_layer)}
head_pair_indices = list(combinations(range(n_head), 2))
head_size = n_embd // n_head
activation_sample_seed = 2026


def get_vram_usage_bytes():
    if device == 'cuda':
        return torch.cuda.memory_allocated()
    return 0

def get_peak_vram_usage_bytes():
    if device == 'cuda':
        return torch.cuda.max_memory_allocated()
    return 0

def reset_peak_vram_usage():
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

def get_gradient_magnitude(model):
    grad_sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq_sum += p.grad.detach().pow(2).sum().item()
    return grad_sq_sum ** 0.5

def write_csv(file_name, rows, fieldnames):
    # Updated to save in the new directory
    file_path = f"{output_dir}/{file_name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def choose_eval_token_indices(global_step, eval_batch_idx):
    rng = random.Random(activation_sample_seed + global_step * 1000003 + eval_batch_idx)
    batch_index = rng.randrange(batch_size)
    token_index = rng.randrange(block_size)
    return batch_index, token_index

def format_vector_for_csv(vec):
    return ' '.join(f"{x:.8f}" for x in vec.tolist())

def collect_head_activation_rows(global_step, eval_batch_idx, X, block_head_outputs):
    batch_index, token_index = choose_eval_token_indices(global_step, eval_batch_idx)
    token_id = int(X[batch_index, token_index].item())
    token_char = itos[token_id]

    for block_idx, block_heads in enumerate(block_head_outputs):
        row = {
            'global_step': global_step,
            'eval_batch_idx': eval_batch_idx,
            'sample_batch_index': batch_index,
            'sample_token_index': token_index,
            'token_id': token_id,
            'token_char': token_char,
        }

        selected_vectors = []
        for head_idx in range(n_head):
            vec = block_heads[head_idx, batch_index, token_index, :].detach().cpu()
            selected_vectors.append(vec)
            row[f'head_{head_idx}_activation'] = format_vector_for_csv(vec)

        for head_i, head_j in head_pair_indices:
            cos_sim = F.cosine_similarity(
                selected_vectors[head_i].unsqueeze(0),
                selected_vectors[head_j].unsqueeze(0),
                dim=-1,
                eps=1e-8,
            ).item()
            row[f'cosine_head_{head_i}_{head_j}'] = cos_sim

        head_activation_rows_by_block[block_idx].append(row)

# --- NEW FUNCTION: Calculate Orthogonality Loss on Value Weights ---
def compute_value_weight_orthogonality_loss(model, margin=0.2):
    """
    Computes the Gram matrix penalty for the Value projection weights across all heads.
    """
    ortho_loss = 0.0
    block_count = 0
    
    for block in model.blocks:
        v_weights = []
        # Extract the Value weights from each head
        for head in block.sa.heads:
            # head.value.weight shape is (head_size, n_embd)
            # We flatten it to treat the entire projection matrix as a single vector
            v_weights.append(head.value.weight.view(-1))
            
        # Stack to shape (n_head, head_size * n_embd)
        W = torch.stack(v_weights)
        
        # 1. L2 Normalize along the feature dimension
        W_norm = F.normalize(W, p=2, dim=1)
        
        # 2. Compute Gram matrix (n_head, n_head)
        gram = torch.matmul(W_norm, W_norm.T)
        
        # 3. Mask out the diagonal (a head's similarity to itself)
        identity_mask = torch.eye(n_head, device=W.device).bool()
        off_diagonals = gram[~identity_mask]
        # if block_count == 0: # Just print for the first block to avoid spam
        #     max_sim = torch.max(torch.abs(off_diagonals)).item()
        #     print(f"Max raw cosine similarity between heads: {max_sim:.4f}")
        # 4. Apply soft margin penalty
        loss = F.relu(torch.abs(off_diagonals) - margin).mean()
        
        ortho_loss += loss
        block_count += 1
        
    return ortho_loss / block_count if block_count > 0 else torch.tensor(0.0, device=device)
# -------------------------------------------------------------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
if not os.path.exists('input.txt'):
    os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(global_step):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            reset_peak_vram_usage()
            step_start_time = time.perf_counter()
            X, Y = get_batch(split)
            if split == 'val':
                logits, loss, block_head_outputs = model(X, Y, return_head_outputs=True)
                collect_head_activation_rows(global_step, k, X, block_head_outputs)
            else:
                logits, loss = model(X, Y)
                
            # Calculate Ortho Loss during eval for tracking
            ortho_loss = compute_value_weight_orthogonality_loss(model, margin=ortho_margin)
            total_loss = loss + (ortho_lambda * ortho_loss)
            
            losses[k] = loss.item()
            step_time_sec = time.perf_counter() - step_start_time
            eval_metrics_rows.append({
                'global_step': global_step,
                'split': split,
                'eval_batch_idx': k,
                'loss': loss.item(),                # Vanilla task loss
                'ortho_loss': ortho_loss.item(),    # Orthogonal penalty loss
                'total_loss': total_loss.item(),    # Combined loss
                'vram_allocated_bytes': get_vram_usage_bytes(),
                'vram_peak_bytes': get_peak_vram_usage_bytes(),
                'time_sec': step_time_sec,
                'gradient_magnitude': float('nan'),
            })
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def collect_eval_hidden_state_magnitudes(num_batches):
    model.eval()
    rows = []
    for batch_idx in range(num_batches):
        X, Y = get_batch('val')
        _, _, hidden_states = model(X, Y, return_hidden_states=True)
        block_magnitudes = [h.abs().mean().item() for h in hidden_states]
        row = {
            'batch_idx': batch_idx,
            'mean_hidden_magnitude': sum(block_magnitudes) / len(block_magnitudes),
        }
        for block_idx, mag in enumerate(block_magnitudes):
            row[f'block_{block_idx}_magnitude'] = mag
        rows.append(row)
    model.train()
    return rows

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_head_outputs=False):
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        if return_head_outputs:
            return out, torch.stack(head_outputs, dim=0)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_head_outputs=False):
        if return_head_outputs:
            sa_out, head_outputs = self.sa(self.ln1(x), return_head_outputs=True)
            x = x + sa_out
        else:
            x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        if return_head_outputs:
            return x, head_outputs
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_hidden_states=False, return_head_outputs=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        hidden_states = []
        block_head_outputs = []
        for block in self.blocks:
            if return_head_outputs:
                x, head_outputs = block(x, return_head_outputs=True)
                block_head_outputs.append(head_outputs)
            else:
                x = block(x)
            if return_hidden_states:
                hidden_states.append(x)
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        if return_hidden_states and return_head_outputs:
            return logits, loss, hidden_states, block_head_outputs
        if return_hidden_states:
            return logits, loss, hidden_states
        if return_head_outputs:
            return logits, loss, block_head_outputs
        return logits, loss

model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(iter)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, ortho_loss (val) {eval_metrics_rows[-1]['ortho_loss']:.4f}, total_loss (val) {eval_metrics_rows[-1]['total_loss']:.4f}")

    reset_peak_vram_usage()
    train_step_start = time.perf_counter()
    xb, yb = get_batch('train')

    # --- MODIFIED TRAINING STEP ---
    logits, loss = model(xb, yb)
    
    # Calculate orthogonality loss
    ortho_loss = compute_value_weight_orthogonality_loss(model, margin=ortho_margin)
    
    # Combine losses
    total_loss = loss + (ortho_lambda * ortho_loss)
    
    optimizer.zero_grad(set_to_none=True)
    # Backward pass on the COMBINED loss
    total_loss.backward()
    grad_magnitude = get_gradient_magnitude(model)
    optimizer.step()
    # ------------------------------
    
    train_step_time_sec = time.perf_counter() - train_step_start
    train_metrics_rows.append({
        'global_step': iter,
        'loss': loss.item(),
        'ortho_loss': ortho_loss.item(),
        'total_loss': total_loss.item(),
        'vram_allocated_bytes': get_vram_usage_bytes(),
        'vram_peak_bytes': get_peak_vram_usage_bytes(),
        'time_sec': train_step_time_sec,
        'gradient_magnitude': grad_magnitude,
    })

# --- UPDATED CSV EXPORTS WITH NEW COLUMNS ---
write_csv(
    'train_metrics.csv',
    train_metrics_rows,
    ['global_step', 'loss', 'ortho_loss', 'total_loss', 'vram_allocated_bytes', 'vram_peak_bytes', 'time_sec', 'gradient_magnitude'],
)

write_csv(
    'eval_metrics.csv',
    eval_metrics_rows,
    ['global_step', 'split', 'eval_batch_idx', 'loss', 'ortho_loss', 'total_loss', 'vram_allocated_bytes', 'vram_peak_bytes', 'time_sec', 'gradient_magnitude'],
)

eval_hidden_rows = collect_eval_hidden_state_magnitudes(eval_iters)
write_csv(
    'eval_hidden_state_magnitudes.csv',
    eval_hidden_rows,
    ['batch_idx'] + [f'block_{i}_magnitude' for i in range(n_layer)] + ['mean_hidden_magnitude'],
)

head_activation_fieldnames = [
    'global_step',
    'eval_batch_idx',
    'sample_batch_index',
    'sample_token_index',
    'token_id',
    'token_char',
] + [
    f'head_{i}_activation' for i in range(n_head)
] + [
    f'cosine_head_{i}_{j}' for i, j in head_pair_indices
]

for block_idx in range(n_layer):
    write_csv(
        f'block{block_idx}_head_activations.csv',
        head_activation_rows_by_block[block_idx],
        head_activation_fieldnames,
    )