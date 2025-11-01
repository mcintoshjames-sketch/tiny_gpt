import os
import math
import time
import requests
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

# WT-103 dataset from Hugging Face (Salesforce/wikitext-103-raw-v1)
WT103_TRAIN_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs/heads/main/wikitext-103-raw-v1/train-00000-of-00001.parquet"
WT103_VALID_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs/heads/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet"
# Fallback: use TinyShakespeare as alternative if HF download fails
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "wt103_train.txt")
VALID_PATH = os.path.join(DATA_DIR, "wt103_valid.txt")
FALLBACK_PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")

# Mac M4 specific: Use best available acceleration
def get_device():
    """
    Select best device for M4 Mac:
    - MPS (Metal Performance Shaders): Uses GPU cores, efficient and recommended
    - CPU: Fallback, will be slower
    
    Note: M4 NPU (Neural Processing Unit) is not directly supported by PyTorch yet.
    MPS via GPU acceleration is the best option for training on M4.
    """
    print("Checking available devices...")
    
    if torch.backends.mps.is_available():
        try:
            # Verify MPS actually works by allocating a small tensor
            test = torch.zeros(1, device="mps")
            del test
            print("✓ Using Apple Metal Performance Shaders (MPS) - GPU acceleration")
            print("  M4 GPU cores will handle tensor operations")
            return torch.device("mps")
        except Exception as e:
            print(f"  MPS available but failed to initialize: {e}")
    
    if torch.cuda.is_available():
        print("✓ Using CUDA")
        return torch.device("cuda")
    
    print("⚠ Falling back to CPU (slower, but will work)")
    print("  Note: If you want faster training, consider using an external GPU")
    return torch.device("cpu")

DEVICE = get_device()

def download_data():
    """Download WT-103 dataset from Salesforce/wikitext-103-raw-v1 using HF datasets library."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if we have valid training data
    if os.path.exists(TRAIN_PATH) and os.path.exists(VALID_PATH):
        size_train = os.path.getsize(TRAIN_PATH)
        size_valid = os.path.getsize(VALID_PATH)
        if size_train > 1000 and size_valid > 1000:  # Basic sanity check
            print(f"✓ Training data already present ({size_train:,} bytes)")
            print(f"✓ Validation data already present ({size_valid:,} bytes)")
            return True
    
    # Try to download WT-103 from Hugging Face using datasets library
    print("Attempting to download WT-103 from Salesforce/wikitext...")
    try:
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library not available")
        
        print("  Loading Salesforce/wikitext dataset...")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        
        print("  Extracting training data...")
        train_texts = dataset["train"]["text"]
        train_text = "\n".join([t for t in train_texts if t and len(t.strip()) > 0])
        
        with open(TRAIN_PATH, "w", encoding="utf-8") as f:
            f.write(train_text)
        print(f"  ✓ Saved training data ({len(train_text):,} bytes)")
        
        print("  Extracting validation data...")
        valid_texts = dataset["validation"]["text"]
        valid_text = "\n".join([t for t in valid_texts if t and len(t.strip()) > 0])
        
        with open(VALID_PATH, "w", encoding="utf-8") as f:
            f.write(valid_text)
        print(f"  ✓ Saved validation data ({len(valid_text):,} bytes)")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to download WT-103: {e}")
    
    # Fallback: use TinyShakespeare
    print("\nFalling back to TinyShakespeare dataset...")
    try:
        print("  Downloading TinyShakespeare...")
        r = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
        r.raise_for_status()
        text = r.text
        
        # Split into train/valid (90/10)
        split_idx = int(0.9 * len(text))
        train_text = text[:split_idx]
        valid_text = text[split_idx:]
        
        with open(TRAIN_PATH, "w", encoding="utf-8") as f:
            f.write(train_text)
        print(f"  ✓ Saved training data ({len(train_text):,} bytes)")
        
        with open(VALID_PATH, "w", encoding="utf-8") as f:
            f.write(valid_text)
        print(f"  ✓ Saved validation data ({len(valid_text):,} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download TinyShakespeare: {e}")
        return False


class CharTokenizer:
    """Character-level tokenizer for easy inspection."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # register buffer for causal mask
        self.register_buffer("mask", torch.tril(torch.ones(2048, 2048)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        return y

def get_batch(data, batch_size, block_size, device):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    # Stack as numpy array first for faster conversion
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=10):
    """Estimate loss on both train and validation sets."""
    model.eval()
    losses = {'train': [], 'val': []}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[split].append(loss.item())
    
    model.train()
    return {k: np.mean(v) for k, v in losses.items()}

def main():
    # Download WT-103
    if not download_data():
        print("Could not download WT-103 dataset. Exiting.")
        return
    
    # Load both datasets first
    print("Loading WT-103 training data...")
    train_text = open(TRAIN_PATH, "r", encoding="utf-8").read()
    print(f"Training text length: {len(train_text):,} characters")
    
    print("Loading WT-103 validation data...")
    valid_text = open(VALID_PATH, "r", encoding="utf-8").read()
    print(f"Validation text length: {len(valid_text):,} characters")
    
    # Build tokenizer from BOTH train and validation data
    # This ensures all characters are in the vocabulary
    print("Building tokenizer from combined vocabulary...")
    combined_text = train_text + valid_text
    tok = CharTokenizer(combined_text)
    print(f"Vocabulary size: {tok.vocab_size}")
    
    # Now encode both with complete vocabulary
    train_data = np.array(tok.encode(train_text), dtype=np.int32)
    val_data = np.array(tok.encode(valid_text), dtype=np.int32)

    # Hyperparameters (small for Mac M4)
    batch_size = 32  # Reduced for MPS memory
    block_size = 128
    n_layer = 3
    n_head = 4
    d_model = 192
    d_ff = 768
    epochs = 5
    iters_per_epoch = 100
    lr = 3e-4
    warmup_iters = 50

    model = TinyGPT(tok.vocab_size, block_size, n_layer, n_head, d_model, d_ff, dropout=0.1).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {DEVICE}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for it in pbar:
            # Warmup
            if global_step < warmup_iters:
                lr_scale = global_step / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale
            
            xb, yb = get_batch(train_data, batch_size, block_size, DEVICE)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            
            if it % 20 == 0:
                losses = estimate_loss(model, train_data, val_data, batch_size, block_size, DEVICE, eval_iters=5)
                pbar.set_postfix(train_loss=f"{losses['train']:.4f}", val_loss=f"{losses['val']:.4f}")

    # Generate sample
    print("\n=== Generating sample ===")
    model.eval()
    start_text = "the"
    start = torch.tensor([tok.encode(start_text)], dtype=torch.long, device=DEVICE)
    out = model.generate(start, max_new_tokens=200, temperature=0.8)
    generated = tok.decode(out[0].tolist())
    print(generated)
    
    # Save checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "tok_itos": tok.itos,
        "tok_stoi": tok.stoi,
        "config": {
            "vocab_size": tok.vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "d_model": d_model,
            "d_ff": d_ff,
        }
    }
    torch.save(checkpoint, "tiny_gpt_checkpoint.pt")
    print("\nCheckpoint saved to tiny_gpt_checkpoint.pt")

if __name__ == "__main__":
    main()
