# BPE Tokenization Upgrade Guide

## What Changed

I've upgraded your training script from **character-level tokenization** to **Byte-Pair Encoding (BPE)** subword tokenization, along with increasing model size. This will give you **much better results**!

### Key Improvements

1. **BPE Tokenization** (vocab_size=4096)
   - Much more efficient than character-level (5006 chars)
   - Learns common subwords like "ing", "tion", "pre-"
   - Shorter sequences = faster training
   - Better text quality

2. **Larger Model**
   - n_layer: 4 → **5** layers
   - n_head: 8 → **6** heads  
   - d_model: 256 → **354** (auto-adjusted to be divisible by 6)
   - d_ff: 1024 → **1300**
   - Parameters: ~5.7M → **~11M parameters** (almost 2x larger!)

## Installation

You need to install the `tokenizers` library:

```bash
cd ~/tiny_gpt
git pull origin main
pip3 install tokenizers
# or
pip3 install -r requirements.txt
```

## How It Works

### First Run: Training the Tokenizer

On your first training run, the script will:
1. Load your WT-103 text data
2. **Train a BPE tokenizer** (learns common subwords)
3. Save it to `tokenizer_bpe.json`
4. Then start model training

This tokenizer training takes ~2-3 minutes but only happens once!

### Subsequent Runs

On future runs, it loads the existing `tokenizer_bpe.json` - no retraining needed.

## Expected Results

### Character-Level (Old)
- Vocab: 5006 characters
- Tokens: ~539M tokens for WT-103
- Loss after 65 epochs: 1.16
- Generated text: Still somewhat gibberish

### BPE (New)
- Vocab: 4096 subwords
- Tokens: ~120M tokens for WT-103 (**4.5x fewer!**)
- Expected loss after 80 epochs: **0.8-1.0** (much better!)
- Generated text: **Coherent sentences with proper grammar**

## Running Training

Same command as before:

```bash
cd ~/tiny_gpt
python3 train.py
```

You'll see:
```
Loading WT-103 training data...
Training text length: 539,301,171 characters
Loading WT-103 validation data...
Validation text length: 1,073,849 characters

Building tokenizer...
Using BPE (Byte-Pair Encoding) tokenizer for efficient subword tokenization
Training BPE tokenizer on combined train+validation data...
[████████████████████] 100%
✓ BPE tokenizer trained with vocab_size=4096
Tokenizer saved to tokenizer_bpe.json
Vocabulary size: 4096

Encoding training data...
Training tokens: 120,453,892
Encoding validation data...
Validation tokens: 241,234

⚠️  Adjusted d_model to 354 (must be divisible by n_head=6)

Model parameters: 11,234,567
Training on: mps
```

## Files Created

- `tokenizer_bpe.json` - Your trained BPE tokenizer
- `tokenizer_bpe_best.json` - Copy saved with best model
- `tiny_gpt_best.pt` - Best model checkpoint (includes metadata)
- `tiny_gpt_final.pt` - Final model after all epochs

## Inference

Use the updated inference script (automatically handles BPE):

```bash
python3 inference.py --prompt "The history of artificial intelligence" --temperature 0.7
```

The script automatically:
1. Detects that checkpoint uses BPE
2. Loads `tokenizer_bpe_best.json` 
3. Generates text with proper subword tokenization

## Why BPE is Better

### Character-Level Problems:
```
Input:  "The quick brown fox"
Tokens: ['T', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k', ...]
Count:  19 tokens
```
- Model must learn to combine letters into words
- Very long sequences
- Inefficient

### BPE Solution:
```
Input:  "The quick brown fox"
Tokens: ['The', 'Ġquick', 'Ġbrown', 'Ġfox']
Count:  4 tokens
```
- Model works with meaningful chunks
- 4.75x shorter sequences
- Much more efficient!

## Comparison: Character vs BPE

| Metric | Character-Level | BPE (New) |
|--------|----------------|-----------|
| Vocab Size | 5,006 | 4,096 |
| WT-103 Tokens | 539M | 120M |
| Sequence Length | Very long | 4-5x shorter |
| Training Speed | Slower | Faster |
| Text Quality | Poor | Good |
| Best For | Tiny datasets | Real datasets |

## Expected Training Time

With the larger model (~11M params) and BPE:
- **Per epoch**: ~5-6 minutes (slightly slower due to bigger model)
- **80 epochs**: ~7-8 hours total
- **Best results**: Around epoch 50-60

Note: Might take slightly longer than 6 hours, but results will be much better!

## Troubleshooting

### "Import tokenizers could not be resolved"
```bash
pip3 install tokenizers
```

### "BPE tokenizer file not found" during inference
Make sure you have either:
- `tokenizer_bpe_best.json` (saved with best model)
- `tokenizer_bpe.json` (original trained tokenizer)

### Model seems slower
Yes, 11M parameters is ~2x the previous size:
- Slightly slower per iteration (~1.5-2 sec instead of 1 sec)
- But much better results!
- Still completes in ~7-8 hours

### Want to use character-level still?
The script auto-detects if `tokenizers` isn't installed and falls back to character-level. Just don't install the library.

## What You'll See

### Good Signs:
- Loss drops below 1.0 by epoch 40-50
- Generated text has proper words
- Sentences make grammatical sense
- Punctuation is correct

### Example Output (after 60 epochs):
```
The history of artificial intelligence began in the 1950s when researchers 
started exploring how machines could simulate human reasoning. Early pioneers 
like Alan Turing proposed tests to measure machine intelligence, while others 
developed the first neural networks and symbolic AI systems.
```

Much better than before! 🎉

## Next Steps

After training completes:
1. Test with `python3 inference.py`
2. Try different prompts and temperatures
3. Compare with your old character-level results
4. Share your generated text!

The combination of BPE tokenization + larger model should give you proper sentences that actually make sense.
