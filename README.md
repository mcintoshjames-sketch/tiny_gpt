````markdown
# Tiny GPT - Educational Language Model

A minimal, from-scratch implementation of a GPT-like transformer for learning purposes.

**Best run on Apple Silicon (M1/M2/M3/M4) with GPU acceleration via MPS!**

## Quick Start (Mac M4)

```bash
# Clone the repository
git clone <repo-url>
cd tiny_gpt

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training (uses MPS for GPU acceleration on M4)
python train.py
```

## Setup Details

### Prerequisites
- **Python 3.9+** (recommend 3.10+)
- **pip**
- **Mac with Apple Silicon** (M1/M2/M3/M4) for GPU acceleration

### Installation

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - `torch` - PyTorch (CPU version, MPS backend enabled automatically)
   - `numpy` - Numerical computing
   - `tqdm` - Progress bars
   - `requests` - HTTP library
   - `datasets` - HuggingFace datasets
   - `pyarrow` - Parquet file support
   - `pandas` - Data manipulation

### Training

```bash
python train.py
```

**What the training script does:**
1. **Downloads WT-103 dataset** from Salesforce/wikitext-103-raw-v1
   - Training: ~516 MB (2.3M lines)
   - Validation: ~1.1 MB
2. **Builds a character-level tokenizer** for easy inspection
3. **Trains a tiny transformer** (3 layers, 4 heads, ~200K parameters)
4. **Saves checkpoint** to `tiny_gpt_checkpoint.pt`
5. **Generates sample text** after training

### Mac M4 GPU Acceleration

The script automatically detects and uses **Metal Performance Shaders (MPS)** for GPU acceleration:

✅ **You'll see ~5-10x speedup vs CPU** on M4

The training output will show:
```
✓ Using Apple Metal Performance Shaders (MPS) - GPU acceleration
  M4 GPU cores will handle tensor operations
```

**Note:** The M4 NPU is not yet directly supported by PyTorch. MPS (GPU acceleration) is the recommended approach.

## Architecture

### Model Components
- **Embeddings:** Token + positional embeddings
- **Transformer blocks:** 3 layers of causal self-attention + feedforward
- **Causal attention mask:** Prevents looking at future tokens
- **Decoder head:** Projects to vocabulary size

### Training Details
- **Tokenizer:** Character-level (vocab size ~100 characters)
- **Loss function:** Cross-entropy on next-token prediction
- **Optimizer:** AdamW with learning rate warmup
- **Evaluation:** Validation loss tracked during training
- **Model size:** ~200K parameters (tiny, good for learning)

## Hyperparameters

Edit these in `train.py` to customize training:

```python
# In main() function:
batch_size = 32          # Batch size (lower if OOM on your Mac)
block_size = 128         # Context window in tokens
n_layer = 3              # Number of transformer layers
n_head = 4               # Number of attention heads
d_model = 192            # Embedding dimension
d_ff = 768               # Feedforward hidden dimension
epochs = 5               # Number of training epochs
iters_per_epoch = 100    # Training iterations per epoch
lr = 3e-4                # Learning rate
```

### Tuning for Your M4

- **Increase batch_size** if you have memory to spare (e.g., 64 or 128)
- **Increase n_layer/n_head/d_model** for a bigger model (~500K parameters still trains fast)
- **Reduce epochs/iters_per_epoch** if you just want quick results

## Dataset

Uses the **Salesforce Wikitext-103 Raw** dataset:
- English Wikipedia text
- Properly formatted and cleaned
- Character-level tokenization works well
- ~540M tokens in training set

### Alternative Datasets

To use a different dataset, modify the `download_data()` function in `train.py`:

```python
# Example: Use TinyShakespeare instead
# Just delete the wt103_train.txt and wt103_valid.txt files
# Script will automatically fallback
```

## Generated Samples

After training, the script generates sample text. Example output:
```
ROMEO: Be not, and be it better than the poor and the...
```

Quality improves with more training!

## Next Steps to Improve

1. **Replace char tokenizer** with BPE (byte-pair encoding) for better efficiency
2. **Add gradient checkpointing** to train larger models
3. **Implement distributed training** with `torch.distributed` (multi-GPU)
4. **Try inference** with the saved checkpoint
5. **Add temperature sampling** for more creative text generation
6. **Implement mixed-precision training** with `torch.autocast` for faster training

## Troubleshooting

### "Module not found" errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Out of Memory (OOM) errors
```python
# Reduce batch size or model size in train.py:
batch_size = 16  # Lower batch size
n_layer = 2      # Fewer layers
d_model = 128    # Smaller embedding dimension
```

### Training is slow
- Verify MPS is being used: Check output at start of `python train.py`
- You should see: `✓ Using Apple Metal Performance Shaders (MPS)`
- If not, file an issue

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Karpathy's MinGPT](https://github.com/karpathy/minGPT) - Educational GPT implementation
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html) - Apple GPU acceleration

````
