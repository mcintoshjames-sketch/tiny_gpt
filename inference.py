"""
Inference script for Tiny GPT
Generate text using a pre-trained checkpoint
"""
import os
import torch
import argparse
from train import TinyGPT, CharTokenizer


def load_checkpoint(checkpoint_path):
    """Load model and tokenizer from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Recreate tokenizer
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.itos = checkpoint["tok_itos"]
    tok.stoi = checkpoint["tok_stoi"]
    tok.vocab_size = len(tok.chars)
    
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
    
    # Decode
    generated_text = tok.decode(output_ids[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Tiny GPT")
    parser.add_argument(
        "--checkpoint",
        default="tiny_gpt_checkpoint.pt",
        help="Path to checkpoint (default: tiny_gpt_checkpoint.pt)",
    )
    parser.add_argument(
        "--prompt",
        default="The",
        help="Prompt to start generation (default: 'The')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
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
        default=3,
        help="Number of samples to generate (default: 3)",
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
    
    print(f"\nModel config: {config}")
    print(f"Vocab size: {config['vocab_size']}")
    
    # Generate samples
    print(f"\n{'='*60}")
    print(f"Generating {args.num_samples} samples")
    print(f"Prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    for i in range(args.num_samples):
        print(f"Sample {i+1}:")
        print("-" * 60)
        generated = generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )
        print(generated)
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
