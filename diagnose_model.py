#!/usr/bin/env python3
"""
Diagnose trained model quality and tokenizer health
"""

import torch
import os
from train import BPETokenizer, TOKENIZERS_AVAILABLE

def diagnose():
    print("=" * 80)
    print("TinyGPT Model Diagnostics")
    print("=" * 80)
    
    # Check checkpoint
    if not os.path.exists("tiny_gpt_best.pt"):
        print("\n❌ No checkpoint found: tiny_gpt_best.pt")
        return
    
    print("\n1. Loading checkpoint...")
    ckpt = torch.load("tiny_gpt_best.pt", map_location="cpu", weights_only=False)
    
    print(f"   ✓ Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"   ✓ Train loss: {ckpt.get('train_loss', 'unknown'):.4f}")
    print(f"   ✓ Val loss: {ckpt.get('val_loss', 'unknown'):.4f}")
    print(f"   ✓ Vocab size: {ckpt['config']['vocab_size']}")
    print(f"   ✓ Model params: {sum(p.numel() for p in ckpt['model_state'].values()):,}")
    
    # Check tokenizer
    if not TOKENIZERS_AVAILABLE:
        print("\n❌ tokenizers library not available")
        return
    
    tokenizer_path = "tokenizer_bpe_best.json"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer_bpe.json"
    
    if not os.path.exists(tokenizer_path):
        print(f"\n❌ No tokenizer found")
        return
    
    print(f"\n2. Testing tokenizer: {tokenizer_path}")
    tok = BPETokenizer(vocab_size=ckpt['config']['vocab_size'])
    tok.load(tokenizer_path)
    
    # Test tokenization quality
    test_cases = [
        "The history of artificial intelligence",
        "Machine learning models",
        "Natural language processing",
        "Neural networks are powerful",
        "Deep learning algorithms"
    ]
    
    print("\n3. Tokenization Quality Test:")
    print("-" * 80)
    
    total_ratio = 0
    for test_text in test_cases:
        tokens = tok.encode(test_text)
        decoded = tok.decode(tokens)
        ratio = len(test_text) / len(tokens)
        total_ratio += ratio
        
        print(f"\nInput:  '{test_text}'")
        print(f"Tokens: {len(tokens)} tokens")
        print(f"Ratio:  {ratio:.2f}x compression")
        print(f"Decoded: '{decoded}'")
        
        # Check if decoding matches
        if decoded.strip().lower() != test_text.strip().lower():
            print("⚠️  Decoded text doesn't match input!")
    
    avg_ratio = total_ratio / len(test_cases)
    print("\n" + "=" * 80)
    print(f"Average compression: {avg_ratio:.2f}x")
    
    if avg_ratio < 1.5:
        print("❌ FAIL: Character-level behavior (compression < 1.5x)")
    elif avg_ratio < 2.5:
        print("⚠️  POOR: Weak subword learning (compression < 2.5x)")
    elif avg_ratio < 3.5:
        print("✓ GOOD: Decent BPE (compression 2.5-3.5x)")
    else:
        print("✅ EXCELLENT: Strong BPE (compression > 3.5x)")
    
    # Show sample tokens
    print("\n4. Sample Token Breakdown:")
    print("-" * 80)
    sample = "The history of artificial intelligence"
    tokens = tok.encode(sample)
    
    print(f"Text: '{sample}'")
    print(f"Token IDs: {tokens[:20]}...")  # First 20 tokens
    
    # Decode individual tokens to see subwords
    print("\nSubword breakdown:")
    for i, token_id in enumerate(tokens[:15]):  # First 15
        subword = tok.decode([token_id])
        print(f"  Token {i}: [{token_id:4d}] = '{subword}'")
    
    print("\n" + "=" * 80)
    print("\n5. Verdict:")
    
    val_loss = ckpt.get('val_loss', float('inf'))
    
    if val_loss > 3.5:
        print("❌ Model quality: POOR (val_loss > 3.5)")
        print("   The model hasn't learned much yet.")
    elif val_loss > 2.0:
        print("⚠️  Model quality: MEDIOCRE (val_loss 2.0-3.5)")
        print("   Some learning, but needs more training.")
    elif val_loss > 1.5:
        print("✓ Model quality: GOOD (val_loss 1.5-2.0)")
        print("   Decent text generation expected.")
    else:
        print("✅ Model quality: EXCELLENT (val_loss < 1.5)")
        print("   High-quality text generation expected.")
    
    if avg_ratio < 2.0:
        print("\n⚠️  WARNING: Tokenizer has poor compression!")
        print("   This will severely limit generation quality.")
        print("   Recommend retraining with fixed tokenizer.")
    
    print("\n" + "=" * 80)
    print("\n6. Generation Test:")
    print("-" * 80)
    print("Testing actual generation quality...")
    
    # Quick generation test
    from train import TinyGPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TinyGPT(
        ckpt['config']['vocab_size'],
        ckpt['config']['block_size'],
        ckpt['config']['n_layer'],
        ckpt['config']['n_head'],
        ckpt['config']['d_model'],
        ckpt['config']['d_ff']
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    test_prompt = "The history of"
    prompt_tokens = tok.encode(test_prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
    
    generated_text = tok.decode(output_ids[0].tolist())
    
    # Show raw output with markers
    print(f"\nPrompt: '{test_prompt}'")
    print(f"Raw output: '{generated_text[:100]}...'")
    
    # Clean up ByteLevel markers for readable output
    clean_text = generated_text.replace('Ġ', ' ').replace('Ċ', '\n').strip()
    print(f"Clean output: '{clean_text[:100]}...'")
    
    # Analyze quality
    if "Ġ" in generated_text or "Ċ" in generated_text:
        print("\n✓ ByteLevel encoding detected (Ġ = space, Ċ = newline)")
        print("  This is normal for proper BPE tokenization!")
        print("  inference.py now automatically cleans these markers for readable output")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose()
