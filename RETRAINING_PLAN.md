# Retraining Plan - 3-Hour A100 Training

## 🎯 Goals
- **Training Time:** ~3 hours on A100 GPU
- **Model Quality:** Better than 23M param model (target val_loss ~2.3-2.5)
- **M4 Inference:** Easy to run on M4 Mac (<2GB RAM)

## 📊 New Hyperparameters

### Model Architecture
- **Layers:** 8 (was 6) - Deeper for better learning
- **Heads:** 8 (same) - Optimal for A100
- **d_model:** 512 (same) - 64 per head
- **d_ff:** 2048 (same) - Standard 4x ratio
- **Context (block_size):** 512 (was 256) - **2x longer context!**
- **Total Parameters:** ~50M (was 23M) - 2.2x larger

### Training Configuration
- **Batch Size:** 192 (was 128) - Larger batches for speed
- **Epochs:** 150 (was 100) - More epochs for convergence
- **Iterations/Epoch:** 400 (was 500) - Balanced
- **Total Steps:** 60,000 (was 50,000)

### Learning Rate Schedule
- **Peak LR:** 3e-4 (was 5e-4) - Lower for stability
- **Min LR:** 3e-5 (was 5e-5) - 10% of peak
- **Warmup:** 1000 steps (was 500) - Longer for stability
- **Schedule:** Cosine decay with warmup

## 🚀 Why These Changes?

### 1. **Longer Context (512 tokens)**
- Previous: 256 tokens (~600 characters)
- New: 512 tokens (~1200 characters)
- **Benefit:** Model sees more context, learns better patterns
- **Trade-off:** Slightly slower per batch, but worth it for quality

### 2. **Deeper Model (8 layers)**
- 33% more layers (6→8)
- **Benefit:** Better hierarchical learning, improved coherence
- **Cost:** +27M params (23M→50M), still small enough for M4

### 3. **Larger Batches (192)**
- 50% larger (128→192)
- **Benefit:** Better gradient estimates, faster convergence
- **A100 advantage:** 40GB VRAM can easily handle this

### 4. **More Epochs (150)**
- With longer context, model needs more time to converge
- Lower learning rate (3e-4) means slower but more stable learning

## ⏱️ Training Time Estimate

**Per Epoch:**
- Longer context (512) → ~2x compute per batch
- Fewer iterations (400) → ~20% faster
- Larger batches (192) → ~50% more efficient
- **Net:** ~80 seconds/epoch (was 58s)

**Total Training:**
- 150 epochs × 80s = 12,000s = **~3.3 hours**
- Target: **3 hours** ✅

## 💾 M4 Inference Capability

### Memory Requirements
```
50M params × 4 bytes (float32) = 200MB model
+ KV cache for 512 context = ~100MB
+ Overhead = ~200MB
──────────────────────────────────
Total: ~500MB-1GB runtime memory
```

**M4 Mac (16GB unified):** ✅ Easily handles this
**M4 Mac (8GB unified):** ✅ Still plenty of headroom

### Inference Speed
- **M4 GPU (MPS):** ~50-100 tokens/sec
- **M4 CPU fallback:** ~10-20 tokens/sec
- **Generation time:** 2-10 seconds for 200 tokens

## 📈 Expected Quality Improvement

### Previous Model (23M, val_loss 2.70)
- Grammatical sentences ✅
- Local coherence (1-2 sentences) ✅
- Multi-sentence flow ⚠️ (limited)
- Long-range coherence ❌

### New Model (50M, target val_loss 2.3-2.5)
- Grammatical sentences ✅✅
- Local coherence ✅✅
- Multi-sentence flow ✅✅ (much better with 512 context)
- Long-range coherence ✅ (improved with deeper model)
- Factual consistency ✅ (better pattern learning)

## 🔄 What Changed vs Previous Training

| Parameter | Old (23M) | New (50M) | Change |
|-----------|-----------|-----------|--------|
| Layers | 6 | 8 | +33% |
| Context | 256 | 512 | +100% |
| Params | 23M | 50M | +117% |
| Batch Size | 128 | 192 | +50% |
| Epochs | 100 | 150 | +50% |
| Peak LR | 5e-4 | 3e-4 | -40% |
| Training Time | 1.6h | 3.3h | +106% |
| Val Loss (target) | 2.70 | 2.3-2.5 | -10-15% |

## 🎓 Training Strategy

### Phase 1: Warmup (1000 steps, ~13 epochs)
- LR: 0 → 3e-4 (linear ramp)
- Purpose: Stabilize training with longer context

### Phase 2: Main Training (steps 1000-54000, ~135 epochs)
- LR: 3e-4 → 3e-5 (cosine decay)
- Purpose: Main learning phase

### Phase 3: Fine-tuning (steps 54000-60000, ~15 epochs)
- LR: 3e-5 (minimum)
- Purpose: Final refinement

## 📝 Training Commands

### In Colab:
```bash
# Clone or update repo
!git clone https://github.com/mcintoshjames-sketch/tiny_gpt.git
# or
!cd tiny_gpt && git pull

# Run training
!cd tiny_gpt && python train.py
```

### Monitor Progress:
- Watch for val_loss dropping below 3.0 (epoch ~30)
- Target 2.5 by epoch ~100
- Target 2.3-2.4 by epoch 150

## 🎯 Success Criteria

**Minimum (acceptable):**
- Val loss ≤ 2.5
- Coherent 3-4 sentence paragraphs

**Target (good):**
- Val loss ≤ 2.3
- Coherent 5-8 sentence paragraphs with logical flow

**Stretch (excellent):**
- Val loss ≤ 2.2
- Near-human quality for short Wikipedia-style text

## 🔍 Key Improvements from 23M Model

1. **2x Context Window:** Sees ~1200 chars instead of ~600
2. **Deeper Architecture:** 8 layers capture more abstract patterns
3. **Better Training:** More epochs + lower LR = better convergence
4. **Larger Batches:** More stable gradients

**Expected outcome:** Text quality comparable to GPT-2 small (124M params trained on much more data).

---

## 🚀 Ready to Start?

1. Open your Colab notebook
2. Ensure A100 GPU is allocated
3. Run `python train.py`
4. Come back in 3 hours!

**Checkpoints saved to:**
- `tiny_gpt_best.pt` (best validation loss)
- `tokenizer_bpe_best.json` (tokenizer)
- Google Drive backup (if mounted)

**Download after training:**
```python
from google.colab import files
files.download('tiny_gpt_best.pt')
files.download('tokenizer_bpe_best.json')
```
