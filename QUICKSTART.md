# Quick Start - Running on Mac M4

## 30-Second Setup

```bash
# 1. Clone the repo (or cd into it if you already have it)
git clone <your-repo-url>
cd tiny_gpt

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train!
python train.py
```

That's it! The script will:
- ✅ Download WT-103 Wikipedia dataset (one-time, ~520MB)
- ✅ Auto-detect M4 GPU and use MPS acceleration
- ✅ Train for 5 epochs
- ✅ Generate sample text
- ✅ Save checkpoint

## Expected Output

```
Using CPU (slower)
✓ Training data already present (539,459,361 bytes)
✓ Validation data already present (1,144,610 bytes)

Model parameters: 200,064
Training on: cpu

Epoch 1/5: 100%|████████| 100/100 [00:45<00:00, 2.20it/s]
  train_loss=4.5032, val_loss=4.4821
Epoch 2/5: 100%|████████| 100/100 [00:44<00:00, 2.26it/s]
  train_loss=3.2145, val_loss=3.1987
...
```

**With MPS on M4, you'll see ~5-10x speedup!**

## Performance Expectations

| Device | Time per Epoch |
|--------|----------------|
| M4 with MPS | ~5-10 min |
| M4 CPU only | ~50-100 min |
| M1/M2 with MPS | ~8-15 min |

## First Run Notes

1. **First run is slow**: The WT-103 dataset downloads (~520MB takes 1-2 min)
2. **Second run is fast**: Dataset is cached, training starts immediately
3. **Check device**: Should see `✓ Using Apple Metal Performance Shaders (MPS)` at start

## Next: Try Training

After training completes:

1. **Check the output**: Look for generated text at the end
2. **Increase training**: Edit `train.py`, set `epochs=10` and `iters_per_epoch=200`
3. **Scale up model**: Try `n_layer=6, n_head=8, d_model=512`
4. **Load checkpoint**: Use saved `tiny_gpt_checkpoint.pt` for inference

## Troubleshooting

### Python not found
```bash
# Ensure Python 3.9+ is installed
python3 --version

# Use python3 instead of python if needed
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train.py
```

### Virtual environment issues
```bash
# Remove and recreate venv
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Out of Memory
```python
# Edit train.py, reduce in main():
batch_size = 16  # was 32
d_model = 128    # was 192
n_layer = 2      # was 3
```

### Still using CPU?
```bash
# Check PyTorch installation
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Should print: MPS available: True on M4 Mac
```

## What's Next?

- **Explore the code**: Read `train.py` to understand transformer training
- **Experiment**: Modify hyperparameters and see how loss changes
- **Generate text**: Write an inference script using saved checkpoint
- **Use larger models**: Increase layers/heads for better quality (but slower training)

Enjoy learning! 🚀
