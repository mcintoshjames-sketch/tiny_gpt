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

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

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
    """Character-level tokenizer (fallback if tokenizers library not available)."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class BPETokenizer:
    """Byte-Pair Encoding (BPE) tokenizer using HuggingFace tokenizers library."""
    def __init__(self, vocab_size=4096):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library not available. Install with: pip install tokenizers")
        
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # Use ByteLevel pre-tokenizer AND decoder for proper byte-level encoding
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self._trained = False
    
    def train(self, texts):
        """Train BPE tokenizer on text data."""
        # Save texts to temporary file for training
        # CRITICAL: BPE needs newline-separated text to learn merges properly
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            if isinstance(texts, list):
                for text in texts:
                    f.write(text + '\n')
                num_lines = len(texts)
            else:
                # Split long text into lines for BPE to learn patterns
                # Use existing line breaks
                lines = [line for line in texts.split('\n') if line.strip()]
                print(f"  Splitting {len(texts):,} chars into {len(lines):,} lines...")
                
                for line in lines:
                    f.write(line + '\n')
                num_lines = len(lines)
            
            temp_path = f.name
        
        print(f"  Wrote {num_lines:,} lines to training file")
        
        try:
            # Configure BPE trainer for proper subword learning
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<unk>"],
                min_frequency=1,  # Learn all tokens (min_frequency=2 was too restrictive)
                show_progress=True,
            )
            
            print(f"  Training BPE on {num_lines:,} lines with vocab_size={self.vocab_size}...")
            self.tokenizer.train([temp_path], trainer)
            self._trained = True
            print(f"✓ BPE tokenizer trained with vocab_size={self.vocab_size}")
            
            # Verify with multiple diverse test cases
            test_texts = [
                "The history of artificial intelligence began in the 1950s.",
                "Machine learning models require large amounts of data.",
                "Natural language processing is a subfield of AI."
            ]
            
            ratios = []
            for test_text in test_texts:
                test_encoding = self.tokenizer.encode(test_text)
                ratio = len(test_text) / len(test_encoding.ids)
                ratios.append(ratio)
            
            avg_ratio = sum(ratios) / len(ratios)
            print(f"✓ Average compression ratio: {avg_ratio:.2f}x")
            
            # Lower threshold for smaller vocab sizes (4096 vocab won't compress as well as 8192)
            min_ratio = 1.5 if self.vocab_size <= 4096 else 2.0
            
            if avg_ratio < min_ratio:
                raise ValueError(
                    f"BPE training produced poor tokenization (compression {avg_ratio:.2f}x < {min_ratio}x). "
                    f"This indicates character-level behavior."
                )
        finally:
            os.unlink(temp_path)
    
    def encode(self, text):
        """Encode text to token IDs."""
        if not self._trained:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, ids):
        """Decode token IDs to text."""
        if not self._trained:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        return self.tokenizer.decode(ids)
    
    def save(self, path):
        """Save tokenizer to disk."""
        self.tokenizer.save(path)
    
    def load(self, path):
        """Load tokenizer from disk."""
        self.tokenizer = Tokenizer.from_file(path)
        self._trained = True

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
    # Convert to long (int64) for PyTorch CUDA compatibility
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
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
    
    # Detect environment early (needed for vocab size and encoding strategy)
    import platform
    is_cloud = 'COLAB_GPU' in os.environ or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or not platform.processor()
    if is_cloud:
        print("✓ Detected cloud environment (Colab/Kaggle)")
        
        # Optional: Mount Google Drive to persist models across sessions
        # This prevents losing trained models when Colab disconnects
        try:
            from google.colab import drive
            drive_mount_point = '/content/drive'
            if not os.path.exists(drive_mount_point):
                print("\n💾 Mounting Google Drive to save models persistently...")
                drive.mount(drive_mount_point)
                print("✓ Google Drive mounted at /content/drive")
                
                # Create backup directory in Google Drive
                backup_dir = os.path.join(drive_mount_point, 'MyDrive', 'tiny_gpt_checkpoints')
                os.makedirs(backup_dir, exist_ok=True)
                print(f"✓ Checkpoint backup directory: {backup_dir}")
            else:
                backup_dir = os.path.join(drive_mount_point, 'MyDrive', 'tiny_gpt_checkpoints')
                print(f"✓ Google Drive already mounted, backups: {backup_dir}")
        except Exception as e:
            print(f"⚠️  Google Drive mount skipped: {e}")
            backup_dir = None
    else:
        print("✓ Detected local environment")
        backup_dir = None
    
    # Check if we should use smaller dataset for faster training
    # Delete WT-103 files and script will use TinyShakespeare fallback
    if not os.path.exists(TRAIN_PATH) or os.path.getsize(TRAIN_PATH) < 10_000_000:
        print("\n⚠️  Using smaller dataset for faster training and better results")
        print("WT-103 is very large for character-level models.")
        print("Consider using TinyShakespeare by deleting the data files.\n")
    
    # Load both datasets first
    print("Loading WT-103 training data...")
    train_text = open(TRAIN_PATH, "r", encoding="utf-8").read()
    print(f"Training text length: {len(train_text):,} characters")
    
    print("Loading WT-103 validation data...")
    valid_text = open(VALID_PATH, "r", encoding="utf-8").read()
    print(f"Validation text length: {len(valid_text):,} characters")
    
    # Build tokenizer - Use BPE for better efficiency
    print("\nBuilding tokenizer...")
    use_bpe = TOKENIZERS_AVAILABLE
    
    if use_bpe:
        print("Using BPE (Byte-Pair Encoding) tokenizer for efficient subword tokenization")
        
        # Adjust vocab size based on environment (Colab has memory constraints during encoding)
        if is_cloud:
            vocab_size = 4096  # Smaller vocab for Colab (faster encoding, less memory)
            print(f"  Cloud environment: using vocab_size={vocab_size} for faster encoding")
        else:
            vocab_size = 8192  # Larger vocab for local with more control
            print(f"  Local environment: using vocab_size={vocab_size}")
        
        # Check if tokenizer already exists
        tokenizer_path = "tokenizer_bpe.json"
        tokenizer_loaded = False
        
        if os.path.exists(tokenizer_path):
            print(f"Found existing tokenizer at {tokenizer_path}")
            try:
                tok = BPETokenizer(vocab_size=vocab_size)
                tok.load(tokenizer_path)
                
                # CRITICAL: Verify this is actually BPE, not character-level
                # Test on diverse sentences to ensure robust subword tokenization
                test_texts = [
                    "The history of artificial intelligence began in the 1950s.",
                    "Machine learning models require extensive training data.",
                    "Natural language processing systems analyze text."
                ]
                
                all_ratios = []
                for test_text in test_texts:
                    test_ids = tok.encode(test_text)
                    ratio = len(test_text) / len(test_ids)
                    all_ratios.append(ratio)
                
                avg_ratio = sum(all_ratios) / len(all_ratios)
                print(f"   Test compression ratio: {avg_ratio:.2f}x")
                
                # Strict threshold: must average > 2.0x compression
                if avg_ratio < 2.0:
                    raise ValueError(f"Tokenizer has poor compression ({avg_ratio:.2f}x), retraining...")
                
                print(f"✓ Loaded existing BPE tokenizer (vocab_size={tok.vocab_size})")
                tokenizer_loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load existing tokenizer: {e}")
                print("   Deleting corrupted tokenizer and training new one...")
                os.remove(tokenizer_path)  # Delete the bad tokenizer file
        
        if not tokenizer_loaded:
            print("Training BPE tokenizer on combined train+validation data...")
            tok = BPETokenizer(vocab_size=vocab_size)
            combined_text = train_text + valid_text
            tok.train(combined_text)
            tok.save(tokenizer_path)
            print(f"✓ Tokenizer saved to {tokenizer_path}")
        
        print(f"Vocabulary size: {tok.vocab_size}")
    else:
        print("⚠️  tokenizers library not available, falling back to character-level")
        print("   Install with: pip install tokenizers")
        print("   Character-level tokenization is slower and less efficient")
        combined_text = train_text + valid_text
        tok = CharTokenizer(combined_text)
        print(f"Vocabulary size: {tok.vocab_size}")
    
    # Encode both datasets - with caching to avoid re-encoding
    train_cache_path = "data/train_tokens.npy"
    val_cache_path = "data/val_tokens.npy"
    
    # Check if cached encoded data exists and matches current tokenizer
    cache_valid = False
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        try:
            # Load cached tokens
            print("\nFound cached encoded data, loading...")
            train_data = np.load(train_cache_path)
            val_data = np.load(val_cache_path)
            print(f"✓ Loaded {len(train_data):,} training tokens from cache")
            print(f"✓ Loaded {len(val_data):,} validation tokens from cache")
            cache_valid = True
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            print("   Will re-encode data...")
    
    if not cache_valid:
        # Encode both datasets - optimized for environment
        print("\nEncoding training data...")
        print(f"  Total characters to encode: {len(train_text):,}")
        
        if is_cloud and use_bpe:
            # Colab/Cloud: Use disk-based encoding to avoid memory limits
            print("  Detected cloud environment - using disk-based encoding (memory-safe)")
            chunk_size = 20_000_000  # 20MB chunks
            num_chunks = (len(train_text) + chunk_size - 1) // chunk_size
            
            print(f"  Processing {num_chunks} chunks of ~{chunk_size/1e6:.0f}MB each...")
            print(f"  First chunk preview: '{train_text[:100]}...'")
            
            import time
            import gc
            import tempfile
            
            # Write tokens to a temporary file instead of keeping in memory
            temp_token_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.npy')
            temp_token_path = temp_token_file.name
            temp_token_file.close()
            
            total_tokens = 0
            
            for chunk_idx in range(num_chunks):
                i = chunk_idx * chunk_size
                chunk = train_text[i:i+chunk_size]
                
                print(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk):,} chars...", end=" ", flush=True)
                start_time = time.time()
                
                try:
                    chunk_tokens = tok.encode(chunk)
                    num_tokens = len(chunk_tokens)
                    total_tokens += num_tokens
                    
                    # Append tokens to file immediately (memory efficient)
                    chunk_array = np.array(chunk_tokens, dtype=np.int32)
                    with open(temp_token_path, 'ab') as f:
                        chunk_array.tofile(f)
                    
                    elapsed = time.time() - start_time
                    print(f"✓ {num_tokens:,} tokens ({elapsed:.1f}s, total: {total_tokens:,})")
                    
                    # Free memory immediately
                    del chunk_tokens, chunk_array, chunk
                    gc.collect()
                    
                except Exception as e:
                    print(f"\n  ❌ Error encoding chunk {chunk_idx+1}: {e}")
                    os.unlink(temp_token_path)
                    raise
            
            # Load all tokens from file at once
            print(f"  Loading {total_tokens:,} tokens from disk...")
            train_data = np.fromfile(temp_token_path, dtype=np.int32)
            os.unlink(temp_token_path)
            print(f"✓ Training tokens: {len(train_data):,}")
            
            print("\nEncoding validation data...")
            val_tokens = tok.encode(valid_text)
            val_data = np.array(val_tokens, dtype=np.int32)
            del val_tokens
            print(f"✓ Validation tokens: {len(val_data):,}")
            
            # Save encoded tokens to cache for future runs
            print("\nSaving encoded tokens to cache...")
            try:
                np.save(train_cache_path, train_data)
                np.save(val_cache_path, val_data)
                print(f"✓ Cached tokens saved to {train_cache_path} and {val_cache_path}")
                print("  (Future runs will skip encoding if cache exists)")
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")
        else:
            # Local Mac: Chunked encoding (memory-safe for M4 with 8-16GB RAM)
            print("  Using chunked encoding for memory safety")
            chunk_size = 10_000_000  # 10MB chunks for local
            train_tokens = []
            num_chunks = (len(train_text) + chunk_size - 1) // chunk_size
            
            with tqdm(total=num_chunks, desc="  Encoding chunks", unit="chunk") as pbar:
                for i in range(0, len(train_text), chunk_size):
                    chunk = train_text[i:i+chunk_size]
                    train_tokens.extend(tok.encode(chunk))
                    pbar.update(1)
            
            train_data = np.array(train_tokens, dtype=np.int32)
            del train_tokens
            print(f"✓ Training tokens: {len(train_data):,}")
            
            print("\nEncoding validation data...")
            val_tokens = tok.encode(valid_text)
            val_data = np.array(val_tokens, dtype=np.int32)
            del val_tokens
            print(f"✓ Validation tokens: {len(val_data):,}")
            
            # Save encoded tokens to cache for future runs
            print("\nSaving encoded tokens to cache...")
            try:
                np.save(train_cache_path, train_data)
                np.save(val_cache_path, val_data)
                print(f"✓ Cached tokens saved to {train_cache_path} and {val_cache_path}")
                print("  (Future runs will skip encoding if cache exists)")
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")

    # Hyperparameters - optimized based on environment
    if is_cloud:
        # A100 GPU optimization: 3-hour training, M4-compatible inference
        # Target: ~50M params, better quality than 23M, still runs on M4
        batch_size = 192  # A100 can handle very large batches (40GB VRAM)
        block_size = 512  # Much longer context for better understanding
        n_layer = 8  # Deeper model for better learning
        n_head = 8  # Optimal for A100 parallelization (64 per head)
        d_model = 512  # Larger embeddings (64 per head)
        d_ff = 2048  # 4x d_model (standard transformer ratio)
        epochs = 150  # More epochs for convergence with longer context
        iters_per_epoch = 400  # Balanced steps per epoch
        lr = 3e-4  # Lower peak LR for stability with longer context
        warmup_iters = 1000  # Longer warmup for larger model + context
        min_lr = 3e-5  # 10% of peak for final fine-tuning
        grad_accum_steps = 1  # No accumulation needed with large batch size
        print("✓ Using A100-optimized hyperparameters (3-hour training target)")
        print(f"  Model: 8 layers, 512 d_model, 512 context → ~50M params")
        print(f"  Estimated training time: ~3 hours for 150 epochs")
        print(f"  M4 inference: ~2GB RAM (easily handled by M4 Mac)")
    else:
        # M4 Mac optimization: Smaller model for MPS/CPU
        batch_size = 64  # Reasonable for M4
        block_size = 128  # Good context window
        n_layer = 5  # Moderate depth
        n_head = 6  # Good attention coverage
        d_model = 360  # 60 per head
        d_ff = 1440  # 4x d_model
        epochs = 80  # Reasonable overnight training
        iters_per_epoch = 400  # Good convergence
        lr = 5e-4  # Peak learning rate
        warmup_iters = 200  # Standard warmup
        min_lr = 5e-5  # 10% of peak
        grad_accum_steps = 2  # Effective batch size = 128
        print("✓ Using M4-optimized hyperparameters")
        print(f"  Estimated params: ~32M, training time: ~6 hours for 80 epochs")
    
    # Ensure d_model is divisible by n_head
    if d_model % n_head != 0:
        # Adjust d_model to nearest multiple of n_head
        d_model = ((d_model // n_head) + 1) * n_head
        print(f"⚠️  Adjusted d_model to {d_model} (must be divisible by n_head={n_head})")

    model = TinyGPT(tok.vocab_size, block_size, n_layer, n_head, d_model, d_ff, dropout=0.1).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {DEVICE}\n")
    
    # A100 optimization: use bfloat16 mixed precision for faster training
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8  # A100 supports bfloat16
    if use_amp:
        print("✓ Using automatic mixed precision (AMP) with bfloat16 for faster training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("✓ Using full precision (float32)")
        scaler = None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Track best validation loss for checkpointing
    best_val_loss = float('inf')
    total_iters = epochs * iters_per_epoch

    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for it in pbar:
            # Learning rate schedule: warmup + cosine decay
            if global_step < warmup_iters:
                # Linear warmup
                lr_scale = global_step / warmup_iters
                current_lr = lr * lr_scale
            else:
                # Cosine annealing from lr to min_lr
                decay_ratio = (global_step - warmup_iters) / (total_iters - warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                current_lr = min_lr + coeff * (lr - min_lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Gradient accumulation with optional mixed precision
            optimizer.zero_grad()
            accum_loss = 0.0
            for micro_step in range(grad_accum_steps):
                xb, yb = get_batch(train_data, batch_size, block_size, DEVICE)
                
                if use_amp:
                    # Mixed precision forward pass
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits, loss = model(xb, yb)
                        loss = loss / grad_accum_steps
                    
                    # Scaled backward pass
                    scaler.scale(loss).backward()
                    accum_loss += loss.item()
                else:
                    # Standard float32 training
                    logits, loss = model(xb, yb)
                    loss = loss / grad_accum_steps
                    loss.backward()
                    accum_loss += loss.item()
            
            # Gradient clipping and optimizer step
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            global_step += 1
            
            # Evaluate and display progress every 100 iterations
            if it % 100 == 0:
                losses = estimate_loss(model, train_data, val_data, batch_size, block_size, DEVICE, eval_iters=10)
                pbar.set_postfix(
                    train_loss=f"{losses['train']:.4f}", 
                    val_loss=f"{losses['val']:.4f}",
                    batch_loss=f"{accum_loss:.4f}",
                    lr=f"{current_lr:.2e}"
                )
        
        # Evaluate at end of each epoch and save best model
        losses = estimate_loss(model, train_data, val_data, batch_size, block_size, DEVICE, eval_iters=20)
        print(f"\nEpoch {epoch+1}/{epochs} complete - Train: {losses['train']:.4f}, Val: {losses['val']:.4f}")
        
        # Save checkpoint if this is the best model so far
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                "model_state": model.state_dict(),
                "tokenizer_type": "bpe" if use_bpe else "char",
                "config": {
                    "vocab_size": tok.vocab_size,
                    "block_size": block_size,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "d_model": d_model,
                    "d_ff": d_ff,
                },
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "epoch": epoch + 1
            }
            
            # For character tokenizer, save vocab mappings
            if not use_bpe:
                checkpoint["tok_itos"] = tok.itos
                checkpoint["tok_stoi"] = tok.stoi
            
            torch.save(checkpoint, "tiny_gpt_best.pt")
            print(f"✓ Saved best model (val_loss={best_val_loss:.4f}) to tiny_gpt_best.pt")
            
            # Also copy the tokenizer file if using BPE
            if use_bpe and os.path.exists("tokenizer_bpe.json"):
                import shutil
                shutil.copy("tokenizer_bpe.json", "tokenizer_bpe_best.json")
                print(f"✓ Saved tokenizer to tokenizer_bpe_best.json")
            
            # Backup to Google Drive if available
            if backup_dir:
                try:
                    # Ensure backup directory exists
                    os.makedirs(backup_dir, exist_ok=True)
                    shutil.copy("tiny_gpt_best.pt", os.path.join(backup_dir, "tiny_gpt_best.pt"))
                    if use_bpe:
                        shutil.copy("tokenizer_bpe_best.json", os.path.join(backup_dir, "tokenizer_bpe_best.json"))
                    print(f"✓ Backed up to Google Drive: {backup_dir}")
                except Exception as e:
                    print(f"⚠️  Google Drive backup failed: {e}")

    # Generate sample
    print("\n=== Generating sample ===")
    model.eval()
    start_text = "the"
    start = torch.tensor([tok.encode(start_text)], dtype=torch.long, device=DEVICE)
    out = model.generate(start, max_new_tokens=200, temperature=0.8)
    generated = tok.decode(out[0].tolist())
    print(generated)
    
    # Save final checkpoint
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    checkpoint = {
        "model_state": model.state_dict(),
        "tokenizer_type": "bpe" if use_bpe else "char",
        "config": {
            "vocab_size": tok.vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "d_model": d_model,
            "d_ff": d_ff,
        }
    }
    
    # For character tokenizer, save vocab mappings
    if not use_bpe:
        checkpoint["tok_itos"] = tok.itos
        checkpoint["tok_stoi"] = tok.stoi
    
    torch.save(checkpoint, "tiny_gpt_final.pt")
    print("Final checkpoint saved to tiny_gpt_final.pt")
    print("Best model saved to tiny_gpt_best.pt")
    
    if use_bpe:
        print("BPE tokenizer saved to tokenizer_bpe.json (best: tokenizer_bpe_best.json)")
    
    print("\n🎉 Training finished! Use tiny_gpt_best.pt for inference.")

if __name__ == "__main__":
    main()
