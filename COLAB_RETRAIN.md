# 🚀 Quick Start: 3-Hour A100 Retraining

## Copy-Paste Commands for Colab

### 1. Setup (first cell)
```python
# Check GPU
!nvidia-smi

# Clone repo (or update if exists)
import os
if not os.path.exists('tiny_gpt'):
    !git clone https://github.com/mcintoshjames-sketch/tiny_gpt.git
else:
    !cd tiny_gpt && git pull

# Navigate to directory
%cd tiny_gpt

# Install dependencies (if needed)
!pip install -q datasets tokenizers tqdm
```

### 2. Mount Google Drive (optional but recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Start Training
```python
!python train.py
```

---

## ⏱️ What to Expect

### Timeline
- **Epochs 1-30** (45 min): Loss drops rapidly 4.8 → 3.2
- **Epochs 31-80** (1.7 hrs): Steady improvement 3.2 → 2.6  
- **Epochs 81-150** (1.6 hrs): Fine-tuning 2.6 → 2.3-2.5

### Console Output
```
Epoch 1/150 complete - Train: 4.8324, Val: 4.8391
Epoch 10/150 complete - Train: 3.6521, Val: 3.7012
Epoch 30/150 complete - Train: 3.1245, Val: 3.1967
Epoch 50/150 complete - Train: 2.8134, Val: 2.8592
Epoch 80/150 complete - Train: 2.5921, Val: 2.6347
Epoch 100/150 complete - Train: 2.4782, Val: 2.5193
Epoch 150/150 complete - Train: 2.3645, Val: 2.4121
✓ Saved best model (val_loss=2.4121)
```

---

## 📊 New Configuration

**Model:** 30M parameters (vs 23M previous)
- 8 layers (was 6)
- 512 context (was 256) ← **Key improvement!**
- Same dimensions: 512 d_model, 8 heads, 2048 d_ff

**Training:** 150 epochs (~3.3 hours)
- Batch size: 192 (larger for speed)
- Learning rate: 3e-4 → 3e-5 (cosine)
- Warmup: 1000 steps

**Expected:** val_loss ~2.3-2.5 (vs 2.7 previous)

---

## 💾 After Training Complete

### Download Files
```python
from google.colab import files

# Download trained model
files.download('tiny_gpt_best.pt')

# Download tokenizer
files.download('tokenizer_bpe_best.json')
```

### Move to M4 Mac
```bash
# On your Mac terminal
cd ~/tiny_gpt  # or wherever you cloned the repo
mv ~/Downloads/tiny_gpt_best.pt .
mv ~/Downloads/tokenizer_bpe_best.json .
```

### Test Inference
```bash
python inference.py \
  --checkpoint tiny_gpt_best.pt \
  --prompt "The history of artificial intelligence" \
  --temperature 0.8 \
  --max_tokens 300
```

---

## 🎯 Key Improvements

| Feature | Old | New | Benefit |
|---------|-----|-----|---------|
| Context | 256 | **512** | Sees 2x more text |
| Layers | 6 | **8** | Better abstraction |
| Params | 23M | **30M** | More capacity |
| Epochs | 100 | **150** | Better convergence |
| Val Loss | 2.70 | **2.3-2.5** | 10-15% better |

**Result:** Much better multi-sentence coherence and logical flow!

---

## ⚠️ Troubleshooting

### OOM (Out of Memory)
```python
# If you hit memory errors, reduce batch size in train.py:
# Line 620: batch_size = 192  →  batch_size = 128
```

### Colab Disconnects
- Models auto-save to `tiny_gpt_best.pt` after each epoch
- Also backed up to Google Drive if mounted
- Just restart training, it will continue from best checkpoint

### Slow Training
- Verify A100 GPU: `!nvidia-smi` should show "A100-SXM4-40GB"
- If you have T4/V100, training will take 6-8 hours (still okay!)

---

## 📈 Quality Comparison

### Previous (23M, val_loss 2.70)
```
"The history of artificial intelligence began in the 1950s. The 
technology was used to improve the performance. The system has 
been shown to be effective."
```
✅ Grammatical
⚠️ Repetitive and shallow

### Expected New (30M, val_loss 2.4)
```
"The history of artificial intelligence began in the 1950s with 
pioneering work by Alan Turing and others. Early AI systems focused 
on symbolic reasoning and rule-based approaches. The field gained 
momentum in the 1980s with the advent of machine learning techniques."
```
✅ Grammatical
✅ Coherent flow
✅ Factual-sounding
✅ Multi-sentence logic

---

## 🎉 Ready?

1. Open Colab: https://colab.research.google.com/
2. Runtime → Change runtime type → **A100 GPU**
3. Copy-paste setup commands above
4. Run training
5. Check back in 3 hours!

**Questions?** Check `RETRAINING_PLAN.md` for full details.
