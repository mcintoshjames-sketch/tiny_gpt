#!/usr/bin/env python3
"""
Quick tokenizer validation test - Run this BEFORE training!
Checks if your BPE tokenizer is working properly or is character-level.
"""

import os
import sys

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    TOKENIZERS_AVAILABLE = True
except ImportError:
    print("❌ tokenizers library not installed!")
    print("   Install with: pip install tokenizers")
    sys.exit(1)

class BPETokenizer:
    """Byte-Pair Encoding (BPE) tokenizer using HuggingFace tokenizers library."""
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self._trained = False
    
    def encode(self, text):
        if not self._trained:
            raise ValueError("Tokenizer not trained yet.")
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, ids):
        if not self._trained:
            raise ValueError("Tokenizer not trained yet.")
        return self.tokenizer.decode(ids)
    
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        self._trained = True

def test_tokenizer(tokenizer_path):
    """Test if tokenizer is proper BPE or character-level."""
    print(f"\n{'='*60}")
    print(f"Testing tokenizer: {tokenizer_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer file not found: {tokenizer_path}")
        print("   You need to train a new tokenizer (it will happen automatically)")
        return False
    
    # Load tokenizer
    print("Loading tokenizer...")
    tok = BPETokenizer(vocab_size=4096)
    try:
        tok.load(tokenizer_path)
        print("✓ Tokenizer loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Test cases
    test_cases = [
        "Hello world, this is a test of tokenization.",
        "The history of artificial intelligence began in the 1950s.",
        "Machine learning models require large amounts of training data.",
        "Natural language processing is a subfield of linguistics and computer science.",
    ]
    
    print("Running compression ratio tests...\n")
    ratios = []
    
    for i, test_text in enumerate(test_cases, 1):
        try:
            test_ids = tok.encode(test_text)
            test_decoded = tok.decode(test_ids)
            
            num_chars = len(test_text)
            num_tokens = len(test_ids)
            compression_ratio = num_chars / num_tokens
            ratios.append(compression_ratio)
            
            print(f"Test {i}:")
            print(f"  Text: '{test_text[:50]}{'...' if len(test_text) > 50 else ''}'")
            print(f"  Characters: {num_chars}")
            print(f"  Tokens: {num_tokens}")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Sample tokens: {test_ids[:10]}...")
            
            # Check if decoded matches original
            if test_decoded.strip() != test_text.strip():
                print(f"  ⚠️  Decoded text doesn't match!")
                print(f"  Original: '{test_text[:30]}'")
                print(f"  Decoded:  '{test_decoded[:30]}'")
            
            print()
        except Exception as e:
            print(f"  ❌ Error encoding test {i}: {e}\n")
            return False
    
    # Calculate average compression ratio
    avg_ratio = sum(ratios) / len(ratios)
    
    print(f"{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average compression ratio: {avg_ratio:.2f}x\n")
    
    # Determine if tokenizer is good
    if avg_ratio < 1.8:
        print("❌ FAIL: Tokenizer is CHARACTER-LEVEL, not BPE!")
        print(f"   Ratio {avg_ratio:.2f}x is too low (expected > 2.0x)")
        print(f"   Your model will generate: 't h e h i s t o r y'")
        print(f"\n   DELETE the tokenizer files:")
        print(f"   rm {tokenizer_path}")
        print(f"   rm tokenizer_bpe_best.json")
        print(f"\n   Then run training to create a proper BPE tokenizer.")
        return False
    elif avg_ratio < 2.5:
        print("⚠️  WARNING: Tokenizer might not be optimal")
        print(f"   Ratio {avg_ratio:.2f}x is borderline (expected 2.5-4.0x)")
        print(f"   It might work, but consider retraining for better quality.")
        return True
    else:
        print("✅ PASS: Tokenizer is proper BPE!")
        print(f"   Ratio {avg_ratio:.2f}x is excellent (2-4 chars per token)")
        print(f"   Your model will generate proper words, not characters")
        print(f"\n   Safe to proceed with training!")
        return True

def main():
    print("\n" + "="*60)
    print("BPE TOKENIZER VALIDATION TEST")
    print("="*60)
    
    tokenizer_files = [
        "tokenizer_bpe.json",
        "tokenizer_bpe_best.json"
    ]
    
    found_any = False
    all_passed = True
    
    for tokenizer_file in tokenizer_files:
        if os.path.exists(tokenizer_file):
            found_any = True
            passed = test_tokenizer(tokenizer_file)
            if not passed:
                all_passed = False
    
    if not found_any:
        print("\n" + "="*60)
        print("NO TOKENIZER FILES FOUND")
        print("="*60)
        print("\n✓ This is good if you're starting fresh!")
        print("  Training will create a new BPE tokenizer automatically.")
        print("\n  You can proceed with: python3 train.py")
        return
    
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if all_passed:
        print("\n✅ All tokenizers are valid BPE tokenizers!")
        print("   Safe to proceed with training.")
    else:
        print("\n❌ Found corrupted/character-level tokenizers!")
        print("   DELETE them before training:")
        print(f"\n   rm tokenizer_bpe.json tokenizer_bpe_best.json")
        print(f"\n   Then run: python3 train.py")

if __name__ == "__main__":
    main()
