# Training on Google Colab (FREE GPU!)

## Why Use Google Colab?

- **FREE T4 GPU access** (no cost!)
- **2-3x faster** than M4 Mac training
- **16GB VRAM** vs M4's shared memory
- **No local resources used** - train while you do other things
- **Persistent storage** via Google Drive

## Quick Start (5 minutes)

### Option 1: Direct Link (Easiest)

1. **Open the notebook**: [Click here to open in Colab](https://colab.research.google.com/github/mcintoshjames-sketch/tiny_gpt/blob/main/train_colab.ipynb)

2. **Enable GPU**:
   - Go to: `Runtime → Change runtime type`
   - Select: `GPU` (T4)
   - Click: `Save`

3. **Run all cells**:
   - Click: `Runtime → Run all`
   - Or press: `Ctrl+F9` (Cmd+F9 on Mac)

4. **Wait ~2-3 hours** for training to complete

5. **Download your model**:
   - Files will auto-download at the end
   - Or manually download from the Files panel (left sidebar)

### Option 2: Manual Upload

1. Go to: https://colab.research.google.com/

2. Click: `File → Upload notebook`

3. Upload: `train_colab.ipynb` from your repository

4. Follow steps 2-5 above

## Expected Training Time

| Hardware | Approximate Time |
|----------|-----------------|
| Google Colab T4 GPU | **2-3 hours** ⚡️ |
| Mac M4 (MPS) | 6-8 hours |
| CPU (any) | 24-48 hours ❌ |

## What Happens During Training

```
✓ Clone repository
✓ Install dependencies (PyTorch, tokenizers, etc.)
✓ Download WikiText-103 (516MB)
✓ Train BPE tokenizer (vocab_size=8192)
✓ Train TinyGPT model (80 epochs, ~13M parameters)
✓ Save best checkpoint (tiny_gpt_best.pt)
✓ Download model to your computer
```

## Colab Limitations & Tips

### Free Tier Limits:
- **12 hours max** per session (plenty for our 2-3 hour training)
- **GPU quota**: ~15-20 hours per week (resets weekly)
- **Storage**: 15GB free on Google Drive

### Pro Tips:

**1. Save to Google Drive** (prevents loss if disconnected):
```python
from google.colab import drive
drive.mount('/content/drive')

# Training will auto-save checkpoints
# Copy to Drive periodically during training
```

**2. Monitor training** with Colab's built-in tools:
- Check GPU usage: `!nvidia-smi`
- Monitor memory: View in "RAM/Disk" indicator (top right)

**3. Keep session alive**:
- Colab may disconnect after ~90 min of inactivity
- Solution: Keep the browser tab open and check occasionally

**4. Resume if interrupted**:
- The notebook checks for existing checkpoints
- Just re-run from the training cell
- It will continue from the last saved epoch

## Using the Trained Model on Your Mac

After Colab downloads your files:

```bash
# 1. Move downloaded files to your project
cd ~/tiny_gpt
mv ~/Downloads/tiny_gpt_best.pt .
mv ~/Downloads/tokenizer_bpe_best.json .

# 2. Test the model
python3 inference.py --prompt "The history of artificial intelligence"

# 3. Generate more samples
python3 inference.py --prompt "Machine learning is" --max_tokens 200 --temperature 0.8
```

## Cost Comparison

| Option | Cost | Training Time | GPU |
|--------|------|---------------|-----|
| **Google Colab Free** | $0 | 2-3 hours | T4 (16GB) ✅ |
| Google Colab Pro | $10/month | 1-2 hours | A100 (40GB) |
| Mac M4 Local | $0 | 6-8 hours | M4 GPU (shared) |
| AWS p3.2xlarge | ~$3/hour | 2-3 hours | V100 |
| RunPod T4 | ~$0.20/hour | 2-3 hours | T4 |

**Winner: Google Colab Free** for this use case! 🏆

## Troubleshooting

### "No GPU available"
- Go to: `Runtime → Change runtime type → GPU (T4)`
- If T4 is unavailable, try later (quota exhausted)

### "Out of memory"
- Reduce `batch_size` in train.py (line 422)
- Change from 64 to 32: `batch_size = 32`

### "Session disconnected"
- Just re-run the notebook
- Existing checkpoints will be loaded automatically

### "Download failed"
- Files may be too large for auto-download
- Alternative: Mount Google Drive and copy files there

## Advanced: Colab Pro

For faster training, consider **Colab Pro** ($10/month):
- A100 GPU (6x faster than T4)
- 40GB VRAM (can train larger models)
- 24-hour sessions (vs 12 hours)
- Priority GPU access

**Training time with A100**: ~45 minutes (vs 2-3 hours on T4)

## Questions?

- Check the [main README](README.md) for model architecture details
- See [LEARNING.md](LEARNING.md) for beginner-friendly explanations
- Run the diagnostics: `python3 diagnose_tokenizer.py`

---

**Ready to train? Click here**: [Open in Google Colab](https://colab.research.google.com/github/mcintoshjames-sketch/tiny_gpt/blob/main/train_colab.ipynb) 🚀
