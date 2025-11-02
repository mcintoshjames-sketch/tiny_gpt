"""
Inference script for Tiny GPT
Generate text using a pre-trained checkpoint
"""
import os
import torch
import argparse
from train import TinyGPT, CharTokenizer, BPETokenizer, TOKENIZERS_AVAILABLE


def load_checkpoint(checkpoint_path):
    """Load model and tokenizer from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Fix for PyTorch 2.6: add weights_only=False for trusted checkpoints
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Determine tokenizer type
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    
    # Recreate tokenizer
    if tokenizer_type == "bpe":
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library required for BPE. Install with: pip install tokenizers")
        
        # Load BPE tokenizer from file
        tokenizer_file = "tokenizer_bpe_best.json"
        if not os.path.exists(tokenizer_file):
            tokenizer_file = "tokenizer_bpe.json"
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"BPE tokenizer file not found: {tokenizer_file}")
        
        tok = BPETokenizer(vocab_size=checkpoint["config"]["vocab_size"])
        tok.load(tokenizer_file)
        print(f"Loaded BPE tokenizer from {tokenizer_file}")
    else:
        # Character-level tokenizer
        tok = CharTokenizer.__new__(CharTokenizer)
        tok.itos = checkpoint["tok_itos"]
        tok.stoi = checkpoint["tok_stoi"]
        tok.vocab_size = len(tok.itos)
        tok.chars = list(tok.itos.values())
        print("Loaded character-level tokenizer")
    
    # Recreate model
    config = checkpoint["config"]
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, tok, config


def generate(model, tok, prompt, max_new_tokens=200, temperature=0.8, device="cpu"):
    """Generate text from a prompt."""
    model = model.to(device)
    
    # Encode prompt
    if len(prompt) == 0:
        prompt = " "
    
    prompt_ids = tok.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    
    # Decode (ByteLevel decoder handles automatic cleanup of markers)
    generated_text = tok.decode(output_ids[0].tolist())
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Tiny GPT")
    parser.add_argument(
        "--checkpoint",
        default="tiny_gpt_best.pt",
        help="Path to checkpoint (default: tiny_gpt_best.pt)",
    )
    parser.add_argument(
        "--prompt",
        default="The history of",
        help="Prompt to start generation (default: 'The history of')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate (default: 300)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling (default: 0.8, range 0.1-2.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tok, config = load_checkpoint(args.checkpoint)
    model = model.to(device)
    
    print(f"\nModel info:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Layers: {config['n_layer']}")
    print(f"  Heads: {config['n_head']}")
    print(f"  d_model: {config['d_model']}")
    
    # Generate samples
    print(f"\n{'='*80}")
    print(f"Generating {args.num_samples} sample(s)")
    print(f"Prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*80}\n")
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"Sample {i+1}:")
            print("-" * 80)
        
        generated = generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )
        print(generated)
        
        if args.num_samples > 1:
            print("-" * 80)
            print()


if __name__ == "__main__":
    main()
