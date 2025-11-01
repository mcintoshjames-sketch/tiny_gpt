# Overnight Training Guide

## What Changed for 6-Hour Training

I've optimized your training script for ~6 hours of training on your M4 Mac. Here's what changed:

### Key Improvements

1. **More Training**: 80 epochs instead of 15 (~6 hours total)
2. **Better Learning**: Cosine learning rate decay from 5e-4 → 5e-5
3. **More Steps**: 400 iterations per epoch (was 300)
4. **Better Regularization**: Dropout increased from 0.1 → 0.15
5. **Smart Checkpointing**: Saves the best model based on validation loss

### What to Expect

**Training Time**: ~6 hours (80 epochs × ~4.5 minutes per epoch)

**Expected Results**:
- Training loss should drop to ~1.0-1.2 (from your current 1.33)
- Generated text should be more coherent with proper words
- Model will learn common phrases and simple grammar

**Files Created**:
- `tiny_gpt_best.pt` - Best model (lowest validation loss) - **Use this for inference!**
- `tiny_gpt_final.pt` - Final model after all 80 epochs

### How to Run Overnight Training

```bash
# Start training (will take ~6 hours)
cd ~/tiny_gpt
python3 train.py

# The script will automatically:
# - Save the best model every time validation improves
# - Show progress every 100 iterations
# - Display learning rate decay
# - Generate a sample at the end
```

### Monitoring Progress

The training will show:
```
Epoch 42/80: 100%|███| 400/400 [02:45<00:00, 2.42it/s, train_loss=1.1234, val_loss=1.2456, batch_loss=1.1100, lr=2.5e-04]
```

Key metrics:
- **train_loss**: Should steadily decrease (lower is better)
- **val_loss**: Should decrease but might plateau (this determines best checkpoint)
- **lr**: Learning rate (starts at 5e-4, decays to 5e-5)

At the end of each epoch:
```
Epoch 42/80 complete - Train: 1.1234, Val: 1.2456
✓ Saved best model (val_loss=1.2456) to tiny_gpt_best.pt
```

### After Training Completes

Generate text with your best model:
```bash
python3 inference.py

# Or with custom prompt:
python3 inference.py --prompt "In the year" --temperature 0.7

# More creative (higher temperature):
python3 inference.py --temperature 1.0

# More conservative (lower temperature):
python3 inference.py --temperature 0.5
```

### What Makes This Better?

1. **Cosine Annealing**: Learning rate gradually decreases, helping the model settle into better solutions
2. **More Epochs**: Character-level models need LOTS of training (you're giving it 5.3x more training)
3. **Better Checkpointing**: Automatically saves the best model, not just the last one
4. **Longer Warmup**: 200 iterations of warmup helps stabilize early training

### Expected Timeline

| Time | Epoch | What's Happening |
|------|-------|------------------|
| 0-15 min | 1-3 | Warmup phase, learning rate ramping up |
| 15-60 min | 4-15 | Rapid learning, loss drops quickly |
| 1-3 hours | 16-40 | Steady improvement, learning common patterns |
| 3-5 hours | 41-70 | Refinement, learning longer sequences |
| 5-6 hours | 71-80 | Fine-tuning, learning rate very low |

### Troubleshooting

**If training seems slow**:
- First iteration takes 10-30 seconds (MPS kernel compilation) - this is normal
- Subsequent iterations should be ~1-2 seconds each
- You should see ~2 iterations/second

**If you need to stop early**:
- Press Ctrl+C to stop
- Your best model is already saved in `tiny_gpt_best.pt`
- You can resume training later (though not from the exact same point)

**If memory issues occur**:
- Reduce `batch_size` from 64 to 32 in train.py
- Reduce `block_size` from 128 to 64 in train.py

### Understanding Your Results

Your previous 15-epoch training:
- Loss: 2.35 → 1.33 (43% improvement)
- Generated: "the settlement of her reduce , oard the most..."
- Quality: Gibberish with some word-like patterns

After 80 epochs, you should see:
- Loss: ~1.0-1.2 (further improvement)
- Generated: More coherent phrases, proper punctuation
- Quality: Not perfect, but recognizable English sentences

Remember: Character-level models need ~10x more training than word/subword models to produce good results. This 80-epoch training is a good balance for learning without spending days training.

### Next Steps After This Training

If results are still not great after 80 epochs:
1. **Switch to subword tokenization** (BPE) - more efficient than character-level
2. **Use smaller dataset** (TinyShakespeare) - easier to overfit and see good results
3. **Increase model size** slightly (more layers or larger d_model)
4. **Train even longer** (150-200 epochs) if you have time

The current setup is optimized for learning how LLMs work. For production-quality models, you'd use:
- Subword tokenization (BPE/WordPiece)
- Much larger models (100M+ parameters)
- Massive datasets (billions of tokens)
- Weeks of training on multiple GPUs
