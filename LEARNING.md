# Tiny GPT: A Beginner's Guide

This guide explains how Large Language Models (LLMs) work from the ground up, assuming no prior knowledge. We'll build up the concepts step by step.

---

## Table of Contents

1. [What is a Language Model?](#what-is-a-language-model)
2. [Understanding Tokenization](#understanding-tokenization)
3. [What are Embeddings?](#what-are-embeddings)
4. [How Neural Networks Learn](#how-neural-networks-learn)
5. [The Transformer Architecture (Simple)](#the-transformer-architecture-simple)
6. [Attention Explained Simply](#attention-explained-simply)
7. [Training Your Model](#training-your-model)
8. [Understanding the Code](#understanding-the-code)
9. [Performance Tuning for M4 Mac](#performance-tuning-for-m4-mac)
10. [Common Questions](#common-questions)

---


---

## What is a Language Model?

**Simple definition**: A language model predicts the next word (or character) given previous words.

### Example:
- Input: "The cat sat on the"
- Model predicts: "mat" (most likely next word)

Think of it like autocomplete on your phone, but much more sophisticated.

### How it works:
1. You give it some text (prompt)
2. It looks at patterns it learned from training data
3. It predicts what comes next
4. Repeat to generate longer text

---

## Understanding Tokenization

**What is tokenization?** Breaking text into small pieces (tokens) that the computer can work with.

### Why do we need it?

Computers don't understand text directly. They only understand numbers. Tokenization converts text → numbers.

### Example with Character-Level Tokenization (what we use):

```
Text:     "hello"
Tokens:   ['h', 'e', 'l', 'l', 'o']
Token IDs: [104, 101, 108, 108, 111]
```

Each unique character gets a unique number (ID).

### Our Tokenizer in Action:

```python
# Build vocabulary from all unique characters in text
chars = ['a', 'b', 'c', ..., 'z', ' ', '.', '!']  # 65 characters for Shakespeare

# Create mapping: character → ID
stoi = {'a': 0, 'b': 1, 'c': 2, ..., 'z': 25, ' ': 26}

# Encode text to numbers
"hello" → [7, 4, 11, 11, 14]

# Decode numbers back to text
[7, 4, 11, 11, 14] → "hello"
```

### Why Character-Level?

**Alternative approaches:**
- **Word-level**: Each word is a token ("hello" = one token)
  - Problem: Vocabulary too large (millions of words)
- **Subword-level** (BPE): Common chunks ("hel" + "lo")
  - Better, but complex to implement
- **Character-level**: Each letter is a token
  - ✅ Small vocabulary (26 letters + punctuation)
  - ✅ Easy to understand
  - ❌ Longer sequences (more computation)

We use character-level because it's simple and educational.

---

## What are Embeddings?

**Simple definition**: Embeddings convert token IDs into dense vectors of numbers that capture meaning.

### The Problem:

Token IDs are just labels (7, 4, 11...). They don't tell the model anything about relationships:
- Is 'h' similar to 'e'?
- Are vowels related?

### The Solution: Embeddings

An embedding is a **list of numbers** (a vector) that represents a token. Similar tokens get similar vectors.

### Visual Example:

```
Character → Token ID → Embedding (vector of numbers)

'a' → 0 → [0.2, -0.5, 0.8, 0.1, ...]  (256 numbers)
'b' → 1 → [0.3, -0.4, 0.7, 0.2, ...]
'e' → 4 → [0.5, -0.1, 0.9, 0.3, ...]  (similar to 'a' - both vowels)
```

### Why Vectors?

Vectors let us do math! Similar meanings = similar vectors:
- Distance between vectors shows similarity
- The model learns these vectors during training

### In Our Code:

```python
# Embedding layer: vocabulary size → embedding dimension
self.tok_emb = nn.Embedding(vocab_size=65, d_model=256)

# When we feed in token ID 7 ('h'):
# Output: A vector of 256 floating point numbers
embedding = self.tok_emb(token_id=7)  # → [0.2, -0.3, 0.8, ..., 0.1]
```

### Two Types of Embeddings We Use:

1. **Token Embeddings**: What character is this?
   - 'a' → [0.2, -0.5, 0.8, ...]

2. **Position Embeddings**: Where is it in the sequence?
   - Position 0 → [0.1, 0.3, -0.2, ...]
   - Position 1 → [0.2, 0.4, -0.1, ...]

We **add them together**:
```
Final embedding = token_embedding + position_embedding
```

This tells the model both **what** the character is and **where** it appears.

---

## How Neural Networks Learn

Think of a neural network as a **function with millions of knobs** (parameters).

### The Learning Process:

1. **Start with random knobs** (random parameters)
2. **Make a prediction** (generate text)
3. **Check how wrong you are** (compute loss)
4. **Adjust knobs slightly** to be less wrong (backpropagation)
5. **Repeat millions of times**

### Example:

```
Correct answer: "The cat sat on the mat"
Model prediction: "The xyz qwr fg the pqr" (terrible at first!)

Loss (error): 4.2 (high = bad)

After 1000 iterations:
Model prediction: "The cat sat on the log"
Loss: 2.1 (lower = better)

After 10000 iterations:
Model prediction: "The cat sat on the mat"
Loss: 0.3 (very low = good!)
```

### What are Parameters?

Every number in the model that gets adjusted during training:
- Embedding weights
- Attention weights
- Layer weights

Our tiny model has **~5 million parameters** (5 million knobs to tune).

### Loss Function:

Measures "how wrong" the model is:
- **High loss** (e.g., 4.0): Model is guessing randomly
- **Low loss** (e.g., 0.5): Model is making good predictions

We use **cross-entropy loss**: compares predicted probabilities vs actual next character.

---

## The Transformer Architecture (Simple)

Our model is a simplified GPT (Generative Pre-trained Transformer). Here's the flow:

```
Input Text: "hello"
      ↓
[1. Tokenize]
      ↓
Token IDs: [7, 4, 11, 11, 14]
      ↓
[2. Embed]
      ↓
Embeddings: Each token → 256-number vector
      ↓
[3. Transformer Blocks] (repeated 4 times)
   ↓
   Attention Layer (look at other tokens)
   ↓
   Feed-Forward Layer (process information)
   ↓
[4. Output Layer]
      ↓
Predictions: Probability for each possible next character
      ↓
[5. Sample]
      ↓
Next character: "!"
```

### Key Components:

1. **Embeddings**: Text → Vectors
2. **Attention**: Let tokens "talk to each other"
3. **Feed-Forward**: Process information
4. **Output**: Vectors → Predictions

---

## Attention Explained Simply

**The core innovation of transformers!**

### The Problem:

When predicting the next word, you need to look at **all previous words**, not just the most recent one.

Example:
```
"The cat, which was very fluffy and orange, sat on the mat"
```

To predict "mat", you need to remember:
- "sat" (most recent)
- "cat" (the subject, far away)
- "on the" (preposition context)

### What is Attention?

**Attention lets each word "look at" and gather information from other words.**

### Simple Analogy:

Imagine you're in a classroom:
- **Query**: "I need help with math"
- **Keys**: Each student says what they're good at
  - Student A: "I'm good at math"
  - Student B: "I'm good at history"
  - Student C: "I'm good at math"
- **Attention scores**: You pay attention to students who match your need
  - Student A: 90% attention
  - Student B: 5% attention
  - Student C: 90% attention
- **Values**: You get information from the students you're attending to

### In Our Model:

Every token creates three things:
1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What information do I have?"
3. **Value (V)**: "Here's my information"

```python
# For the word "sat" trying to predict next word:

Q_sat = "I'm a verb, looking for related context"
K_cat = "I'm a noun, the subject"
K_fluffy = "I'm an adjective"
K_on = "I'm a preposition indicating location"

# Attention scores (how much to pay attention):
score(sat, cat) = 0.8  # High! Subject is important
score(sat, fluffy) = 0.1  # Low, less relevant
score(sat, on) = 0.9  # Very high! Location preposition

# Gather information (weighted by scores):
output_sat = 0.8 × V_cat + 0.1 × V_fluffy + 0.9 × V_on + ...
```

### Multi-Head Attention:

Instead of one attention mechanism, we use **multiple "heads"** that look for different patterns:
- Head 1: Grammar relationships
- Head 2: Semantic meaning
- Head 3: Long-range dependencies
- Head 4: Local context

Each head learns different things!

### Causal Masking (Very Important!):

**Rule**: You can only look at **previous** tokens, not future ones.

```
Predicting character 3:
Can see: [char0, char1, char2] ✅
Cannot see: [char4, char5, ...] ❌
```

This is why it's called "causal" - cause comes before effect.

We use a **mask** to block future tokens:
```
Attention matrix:
     c0  c1  c2  c3  c4
c0 [ ✓   ✗   ✗   ✗   ✗ ]
c1 [ ✓   ✓   ✗   ✗   ✗ ]
c2 [ ✓   ✓   ✓   ✗   ✗ ]
c3 [ ✓   ✓   ✓   ✓   ✗ ]
c4 [ ✓   ✓   ✓   ✓   ✓ ]

✓ = can attend to
✗ = blocked (masked)
```

---

## Training Your Model

### What Happens During Training:

```python
for epoch in range(20):  # Go through dataset 20 times
    for batch in dataset:
        # 1. Get a chunk of text
        input_text = "hello worl"
        target_text = "ello world"  # Shifted by 1 character
        
        # 2. Convert to token IDs
        input_ids = [7, 4, 11, 11, 14, 26, 22, 14, 17, 11]
        target_ids = [4, 11, 11, 14, 26, 22, 14, 17, 11, 3]
        
        # 3. Model makes prediction
        predictions = model(input_ids)
        
        # 4. Compute loss (how wrong?)
        loss = cross_entropy(predictions, target_ids)
        
        # 5. Adjust model parameters to reduce loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
```

### Key Training Concepts:

**Batch Size**: How many examples to process at once
- Larger = faster training (parallel processing)
- Larger = more memory
- We use 64 on M4 Mac

**Learning Rate**: How big the parameter adjustments are
- Too high: Model doesn't learn (jumps around)
- Too low: Training takes forever
- We use 5e-4 (0.0005)

**Epochs**: How many times to go through the entire dataset
- More epochs = better learning (up to a point)
- Too many = overfitting (memorizing, not generalizing)

**Gradient Accumulation**: Trick to simulate larger batches
- Process 2 small batches
- Accumulate gradients
- Update parameters once
- Effective batch size = 64 × 2 = 128

---

## Understanding the Code

Let's walk through the main pieces:

### 1. Tokenizer (Simple Version)

```python
class CharTokenizer:
    def __init__(self, text):
        # Find all unique characters
        self.chars = sorted(list(set(text)))  # ['a', 'b', 'c', ...]
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char → ID
        self.itos = {i: ch for ch, i in self.stoi.items()}      # ID → char
    
    def encode(self, text):
        # "hello" → [7, 4, 11, 11, 14]
        return [self.stoi[c] for c in text]
    
    def decode(self, ids):
        # [7, 4, 11, 11, 14] → "hello"
        return "".join(self.itos[i] for i in ids)
```

### 2. Attention Layer (Simplified Explanation)

```python
class CausalSelfAttention:
    def __init__(self, d_model, n_head):
        # Create Q, K, V projections
        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, d_model)
        
        # 1. Create queries, keys, values
        Q = self.query(x)  # "What am I looking for?"
        K = self.key(x)    # "What info do I have?"
        V = self.value(x)  # "Here's my info"
        
        # 2. Compute attention scores
        # How much should each token attend to each other token?
        scores = Q @ K.transpose()  # Matrix multiplication
        scores = scores / sqrt(d_model)  # Scale
        
        # 3. Apply causal mask (can't see future)
        scores = mask_future(scores)
        
        # 4. Convert to probabilities
        attention_weights = softmax(scores)
        
        # 5. Gather information
        output = attention_weights @ V
        
        return output
```

### 3. Full Model Structure

```python
class TinyGPT:
    def __init__(self):
        # 1. Convert tokens to vectors
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_length, d_model)
        
        # 2. Stack of transformer blocks
        self.blocks = [
            TransformerBlock(),  # Attention + FeedForward
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
        ]
        
        # 3. Output layer: vectors → probabilities
        self.output = Linear(d_model, vocab_size)
    
    def forward(self, token_ids):
        # Get embeddings
        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Get predictions
        logits = self.output(x)
        return logits  # Probability for each token
```

### 4. Generation (How to Create Text)

```python
def generate(prompt, max_tokens=100):
    # Start with prompt
    tokens = tokenizer.encode(prompt)  # "Hello" → [7, 4, 11, 11, 14]
    
    for _ in range(max_tokens):
        # 1. Get model prediction
        logits = model(tokens)
        
        # 2. Get probabilities for next token
        probs = softmax(logits[-1])  # Last position
        
        # 3. Sample next token
        next_token = sample(probs)  # Weighted random choice
        
        # 4. Add to sequence
        tokens.append(next_token)
    
    # Convert back to text
    return tokenizer.decode(tokens)
```

---

## Performance Tuning for M4 Mac

### Challenge: Balancing Quality vs Speed

Training a language model involves many tradeoffs. Here's how we optimized for the M4 Mac:

### 1. Device Selection: MPS (Metal Performance Shaders)

```python
def get_device():
    if torch.backends.mps.is_available():
        # Use Apple's GPU via Metal
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

**Why MPS?**
- M4 has powerful GPU cores that accelerate tensor operations
- 5-10x faster than CPU for matrix multiplications
- PyTorch 2.0+ has good MPS backend support
- Note: M4 NPU (Neural Engine) isn't directly supported by PyTorch yet

### 2. Model Size Optimization

We went through several iterations:

| Version | Params | Speed (it/s) | Quality | Issue |
|---------|--------|--------------|---------|-------|
| Initial | 200K | ~5 | Poor | Too small for WT-103 |
| Large | 24M | ~0.05 | ? | Too slow, appeared frozen |
| **Final** | **~5M** | **~1-2** | **Good** | ✅ Balanced |

**Final Configuration:**
```python
n_layer = 4        # Layers: Each adds depth but slows training
n_head = 8         # Heads: More = better attention, but diminishing returns
d_model = 256      # Embedding dim: Bigger = more capacity, more memory
d_ff = 1024        # Feedforward: Usually 4x d_model
block_size = 128   # Context: Longer = better but O(n²) attention cost
```

**Parameter Count Calculation:**
```
Embeddings: vocab_size * d_model = 5006 * 256 = 1.28M
Position: block_size * d_model = 128 * 256 = 33K
Each Transformer Block: ~4 * d_model² + 2 * d_model * d_ff = ~786K
4 Blocks: 4 * 786K = 3.14M
Output Head: d_model * vocab_size = 256 * 5006 = 1.28M

Total: ~5.7M parameters
```

### 3. Batch Size Tuning

```python
batch_size = 64  # Number of sequences processed in parallel
```

**Tradeoff:**
- **Larger batch** → Better GPU utilization, more stable gradients
- **Smaller batch** → Less memory, faster iteration, more noisy gradients

**Why 64?**
- Fits comfortably in M4 memory (~16GB unified)
- Good GPU utilization
- Not so large that it slows down each iteration

### 4. Gradient Accumulation

```python
grad_accum_steps = 2  # Effective batch size = 64 * 2 = 128
```

**How it works:**
```python
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    xb, yb = get_batch(train_data, batch_size, block_size, DEVICE)
    loss = model(xb, yb) / grad_accum_steps  # Scale loss
    loss.backward()  # Accumulate gradients
optimizer.step()  # Update once after all micro-batches
```

**Benefits:**
- Simulate larger batch size (128) without OOM
- Better gradient estimates
- Minimal memory overhead

### 5. Context Window (block_size)

```python
block_size = 128  # Number of tokens the model sees at once
```

**Attention Complexity: O(n²)**

| Context | Attention Cost | Memory |
|---------|----------------|--------|
| 64 | 4,096 | Low |
| 128 | 16,384 | Medium |
| 256 | 65,536 | High |
| 512 | 262,144 | Very High |

**Why 128?**
- Good balance for character-level (roughly 1-2 sentences)
- Attention cost is manageable
- Fits in memory with other tensors

### 6. Training Duration

```python
epochs = 15           # Full passes through dataset
iters_per_epoch = 300 # Training steps per epoch
# Total: 15 * 300 = 4,500 batches
```

**Calculation for WT-103:**
```
Dataset size: 539M characters
Tokens per batch: 64 * 128 = 8,192
Total tokens per epoch: 300 * 8,192 = 2.46M tokens
Coverage per epoch: 2.46M / 539M = 0.46%

After 15 epochs: 15 * 2.46M = 36.9M tokens seen (~7% of dataset)
```

**Why this matters:**
- Character-level models need to see more examples
- WT-103 is very large (539M characters)
- We're training on a subset for speed
- For production: train for 50+ epochs or use subword tokenization

### 7. Learning Rate Schedule

```python
lr = 5e-4              # Peak learning rate
warmup_iters = 100     # Gradually increase LR at start

# Warmup implementation
if global_step < warmup_iters:
    lr_scale = global_step / warmup_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_scale
```

**Why warmup?**
- Prevents large updates early when model is random
- More stable training
- Common practice for transformers

**Learning Rate Choice:**
- Too high: Training diverges
- Too low: Learns too slowly
- `5e-4` is a good default for AdamW

### 8. Memory Optimization: NumPy Stack

```python
# Before (slow, warning message)
x = [data[i:i+block_size] for i in ix]
x = torch.tensor(x, device=device)

# After (fast, no warning)
x = np.stack([data[i:i+block_size] for i in ix])
x = torch.from_numpy(x).to(device)
```

**Why?**
- Creating tensors from lists of arrays is slow
- Stack into single NumPy array first
- Then convert to PyTorch tensor
- ~2x faster data loading

---

## Common Questions

### Q: Why is my generated text gibberish at first?

**A**: The model starts with random parameters! It takes time to learn patterns. Loss should decrease over epochs:
- Epoch 1: Loss ~4.0, text is random
- Epoch 5: Loss ~2.5, some patterns emerge
- Epoch 15: Loss ~1.8, readable text

### Q: Why character-level instead of words?

**A**: 
- ✅ Simple vocabulary (~65 chars vs millions of words)
- ✅ No out-of-vocabulary problems
- ✅ Can generate any word, including new ones
- ❌ Longer sequences (more computation)

For learning, character-level is perfect!

### Q: What's the difference between training and inference?

**Training**: Adjusting parameters to minimize loss
- Requires: input + target (correct answer)
- Backpropagation, gradients, optimizer
- Slow (hours)

**Inference**: Using trained model to generate
- Requires: only input (prompt)
- No backpropagation
- Fast (seconds)

### Q: Why does it use so much memory?

**Main memory users:**
1. **Model parameters**: 5M parameters × 4 bytes = 20MB
2. **Activations**: Intermediate values during forward pass
3. **Gradients**: Derivatives for backprop
4. **Optimizer state**: Adam stores momentum for each parameter

Batch size × sequence length has huge impact!

### Q: What is MPS?

**MPS (Metal Performance Shaders)**: Apple's GPU acceleration framework
- Uses M4 GPU (not NPU)
- 5-10× faster than CPU for deep learning
- PyTorch supports it natively on Mac

### Q: Can I use the NPU instead of GPU?

**A**: No, not directly with PyTorch. The M4's Neural Engine (NPU) is only accessible through Core ML and only for inference, not training. For training, MPS (GPU) is your best option.

### Q: How do I know if training is working?

**Good signs:**
- Loss decreases over time
- Validation loss follows training loss
- Generated text improves quality
- No NaN or Inf in loss

**Bad signs:**
- Loss doesn't decrease (learning rate too low?)
- Loss explodes (learning rate too high?)
- Validation loss increases while training loss decreases (overfitting)

### Q: Why does the first iteration take so long?

**A**: MPS compiles GPU kernels on first use. This can take 10-30 seconds. Subsequent iterations are much faster (~1 second each).

---

## Key Parameters Explained

### Model Size Parameters:

**d_model (embedding dimension)**: 256
- Size of vectors representing each token
- Larger = more capacity to capture nuances
- Smaller = faster, less memory

**n_layer (number of layers)**: 4
- How many transformer blocks to stack
- More layers = deeper understanding
- But: more computation

**n_head (number of attention heads)**: 8
- How many different attention patterns to learn
- More heads = capture more relationships
- Must divide d_model evenly

**d_ff (feedforward dimension)**: 1024
- Size of intermediate layer in feedforward network
- Usually 4× d_model

**block_size (context window)**: 128
- How many tokens to look at at once
- Larger = better long-range understanding
- But: O(n²) complexity (gets expensive fast!)

### Training Parameters:

**batch_size**: 64
- How many sequences to process in parallel
- Tuned for M4 Mac memory

**learning_rate**: 5e-4
- How much to adjust parameters each step
- Standard for transformers

**epochs**: 15
- How many times to go through dataset
- Balance between learning and overfitting

**gradient_accumulation_steps**: 2
- Simulate larger batch (64 × 2 = 128 effective)
- Better gradients without memory issues

---

## Next Steps

Now that you understand the basics:

1. **Run the training** and watch the loss decrease
2. **Experiment with hyperparameters**:
   - Change `n_layer` (2, 4, 6)
   - Change `d_model` (128, 256, 512)
   - Change `batch_size`
3. **Try different datasets** (Shakespeare vs WT-103)
4. **Modify the code**:
   - Add learning rate scheduling
   - Try different optimizers
   - Implement beam search for generation
5. **Generate text** with `inference.py` after training

---

## Glossary

**Attention**: Mechanism allowing tokens to gather information from other tokens  
**Backpropagation**: Algorithm for computing gradients (how to adjust parameters)  
**Batch**: Group of examples processed together  
**Causal**: Only looking at past, not future (important for text generation)  
**Cross-entropy**: Loss function measuring prediction quality  
**Embedding**: Converting tokens to dense vectors  
**Epoch**: One complete pass through the training dataset  
**Gradient**: Direction and magnitude to adjust a parameter  
**Hyperparameter**: Setting you choose (not learned), like learning rate  
**Inference**: Using a trained model to make predictions  
**Layer**: A processing step in the neural network  
**Loss**: Measure of how wrong the model's predictions are  
**MPS**: Metal Performance Shaders (Apple GPU framework)  
**NPU**: Neural Processing Unit (Apple Neural Engine - not usable for training in PyTorch)  
**Optimizer**: Algorithm for updating parameters (we use AdamW)  
**Parameter**: Learnable number in the model (weight or bias)  
**Softmax**: Converts numbers to probabilities (sum to 1)  
**Token**: Basic unit (in our case, a character)  
**Tokenization**: Converting text to tokens  
**Transformer**: Neural network architecture using attention  
**Vocabulary**: Set of all possible tokens  

---

## Helpful Resources

**Beginner-Friendly:**
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

**Interactive:**
- [Transformer Explainer (Visual)](https://poloclub.github.io/transformer-explainer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

**Original Papers (Advanced):**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

**Questions?** Read through this guide slowly, run the code, and experiment! The best way to learn is by doing. 🚀

Our Tiny GPT is a **decoder-only transformer** (like GPT-2/3/4) that predicts the next character in a sequence.

### High-Level Components

```
Input Text → Tokenizer → Embeddings → Transformer Blocks → Output Head → Next Token
```

### Model Structure

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, d_ff, dropout):
        # 1. Token Embeddings: Convert token IDs to dense vectors
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # 2. Position Embeddings: Add position information
        self.pos_emb = nn.Embedding(block_size, d_model)
        
        # 3. Transformer Blocks: Stack of attention + feedforward layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) 
            for _ in range(n_layer)
        ])
        
        # 4. Layer Norm: Normalize outputs
        self.ln_f = nn.LayerNorm(d_model)
        
        # 5. Output Head: Project back to vocabulary
        self.head = nn.Linear(d_model, vocab_size)
```

**Key Parameters:**
- `vocab_size`: Number of unique tokens (5006 characters in WT-103)
- `block_size`: Context window length (128 tokens)
- `n_layer`: Number of transformer blocks (4 layers)
- `n_head`: Number of attention heads per block (8 heads)
- `d_model`: Embedding dimension (256)
- `d_ff`: Feedforward hidden dimension (1024)

---

## The Attention Mechanism

### What is Attention?

**Attention allows the model to focus on relevant parts of the input when predicting the next token.** Instead of treating all previous tokens equally, the model learns which tokens are most important for the current prediction.

### Causal Self-Attention

We use **causal (masked) self-attention** which ensures the model can only "see" previous tokens, not future ones. This is critical for autoregressive language modeling.

### Mathematical Formulation

Given input sequence embeddings `X` of shape `(batch, seq_len, d_model)`:

1. **Compute Query, Key, Value projections:**
   ```
   Q = X @ W_q    # (batch, seq_len, d_model)
   K = X @ W_k    # (batch, seq_len, d_model)
   V = X @ W_v    # (batch, seq_len, d_model)
   ```

2. **Split into multiple heads:**
   ```
   Q → (batch, n_head, seq_len, d_head)  where d_head = d_model / n_head
   K → (batch, n_head, seq_len, d_head)
   V → (batch, n_head, seq_len, d_head)
   ```

3. **Compute attention scores:**
   ```
   scores = (Q @ K.T) / sqrt(d_head)    # (batch, n_head, seq_len, seq_len)
   ```
   
   The `1/sqrt(d_head)` scaling prevents scores from becoming too large.

4. **Apply causal mask:**
   ```
   mask = [[1, 0, 0],      # Token 0 can only see itself
           [1, 1, 0],      # Token 1 can see tokens 0,1
           [1, 1, 1]]      # Token 2 can see tokens 0,1,2
   
   scores = scores.masked_fill(mask == 0, -inf)  # Force future = -infinity
   ```

5. **Softmax to get attention weights:**
   ```
   attention_weights = softmax(scores, dim=-1)  # (batch, n_head, seq_len, seq_len)
   ```

6. **Weighted sum of values:**
   ```
   output = attention_weights @ V    # (batch, n_head, seq_len, d_head)
   ```

7. **Concatenate heads and project:**
   ```
   output = concat(all_heads)        # (batch, seq_len, d_model)
   output = output @ W_o             # Final projection
   ```

### Why Multiple Heads?

Each attention head can learn different patterns:
- Head 1: Subject-verb agreement
- Head 2: Punctuation patterns
- Head 3: Word associations
- Head 4-8: Other linguistic patterns

This parallel attention allows the model to capture multiple relationships simultaneously.

---

## Implementation Details

### 1. Causal Self-Attention Layer

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: lower triangular matrix
        self.register_buffer("mask", torch.tril(torch.ones(2048, 2048)))

    def forward(self, x):
        B, T, C = x.size()  # batch, seq_len, d_model
        
        # Compute Q, K, V and split into multiple heads
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # Shape: (B, n_head, T, d_head)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Shape: (B, n_head, T, T)
        
        # Apply causal mask
        mask = self.mask[:T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, n_head, T, d_head)
        
        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        
        return y
```

**Key Implementation Details:**

- **`.view()` and `.transpose()`**: Reshape tensors to split/merge attention heads
- **`register_buffer("mask", ...)`**: Stores mask as part of model but not a trainable parameter
- **`masked_fill(mask == 0, -inf)`**: Sets future positions to negative infinity so softmax → 0
- **Scaling by `1/sqrt(d_head)`**: Prevents vanishing gradients in softmax

### 2. Transformer Block

Each transformer block combines attention with a feedforward network:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm architecture (normalize before sublayers)
        x = x + self.attn(self.ln1(x))    # Attention with residual connection
        x = x + self.ff(self.ln2(x))      # Feedforward with residual connection
        return x
```

**Design Patterns:**

- **Residual Connections** (`x = x + ...`): Allow gradients to flow directly through the network
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Pre-Norm** (normalize before sublayer): More stable than post-norm for deep networks
- **GELU Activation**: Smooth activation function that works well for transformers

### 3. Character-Level Tokenizer

```python
class CharTokenizer:
    def __init__(self, text):
        # Get unique characters and sort for consistency
        self.chars = sorted(list(set(text)))
        
        # Create mappings: character ↔ integer ID
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string to int
        self.itos = {i: ch for ch, i in self.stoi.items()}      # int to string
        self.vocab_size = len(self.chars)
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
```

**Why Character-Level?**

✅ **Pros:**
- Simple to understand
- No out-of-vocabulary issues
- Works for any language

❌ **Cons:**
- Very long sequences (every character is a token)
- Harder to learn word-level patterns
- Larger vocabulary for Unicode text

**Better Alternative:** Byte-Pair Encoding (BPE) or SentencePiece for production models.

---

## Performance Tuning for M4 Mac

### Challenge: Balancing Quality vs Speed

Training a language model involves many tradeoffs. Here's how we optimized for the M4 Mac:

### 1. Device Selection: MPS (Metal Performance Shaders)

```python
def get_device():
    if torch.backends.mps.is_available():
        # Use Apple's GPU via Metal
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

**Why MPS?**
- M4 has powerful GPU cores that accelerate tensor operations
- 5-10x faster than CPU for matrix multiplications
- PyTorch 2.0+ has good MPS backend support
- Note: M4 NPU isn't directly supported by PyTorch yet

### 2. Model Size Optimization

We went through several iterations:

| Version | Params | Speed (it/s) | Quality | Issue |
|---------|--------|--------------|---------|-------|
| Initial | 200K | ~5 | Poor | Too small for WT-103 |
| Large | 24M | ~0.05 | ? | Too slow, appeared frozen |
| **Final** | **~5M** | **~1-2** | **Good** | ✅ Balanced |

**Final Configuration:**
```python
n_layer = 4        # Layers: Each adds depth but slows training
n_head = 8         # Heads: More = better attention, but diminishing returns
d_model = 256      # Embedding dim: Bigger = more capacity, more memory
d_ff = 1024        # Feedforward: Usually 4x d_model
block_size = 128   # Context: Longer = better but O(n²) attention cost
```

**Parameter Count Calculation:**
```
Embeddings: vocab_size * d_model = 5006 * 256 = 1.28M
Position: block_size * d_model = 128 * 256 = 33K
Each Transformer Block: ~4 * d_model² + 2 * d_model * d_ff = ~786K
4 Blocks: 4 * 786K = 3.14M
Output Head: d_model * vocab_size = 256 * 5006 = 1.28M

Total: ~5.7M parameters
```

### 3. Batch Size Tuning

```python
batch_size = 64  # Number of sequences processed in parallel
```

**Tradeoff:**
- **Larger batch** → Better GPU utilization, more stable gradients
- **Smaller batch** → Less memory, faster iteration, more noisy gradients

**Why 64?**
- Fits comfortably in M4 memory (~16GB unified)
- Good GPU utilization
- Not so large that it slows down each iteration

### 4. Gradient Accumulation

```python
grad_accum_steps = 2  # Effective batch size = 64 * 2 = 128
```

**How it works:**
```python
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    xb, yb = get_batch(train_data, batch_size, block_size, DEVICE)
    loss = model(xb, yb) / grad_accum_steps  # Scale loss
    loss.backward()  # Accumulate gradients
optimizer.step()  # Update once after all micro-batches
```

**Benefits:**
- Simulate larger batch size (128) without OOM
- Better gradient estimates
- Minimal memory overhead

### 5. Context Window (block_size)

```python
block_size = 128  # Number of tokens the model sees at once
```

**Attention Complexity: O(n²)**

| Context | Attention Cost | Memory |
|---------|----------------|--------|
| 64 | 4,096 | Low |
| 128 | 16,384 | Medium |
| 256 | 65,536 | High |
| 512 | 262,144 | Very High |

**Why 128?**
- Good balance for character-level (roughly 1-2 sentences)
- Attention cost is manageable
- Fits in memory with other tensors

### 6. Training Duration

```python
epochs = 15           # Full passes through dataset
iters_per_epoch = 300 # Training steps per epoch
# Total: 15 * 300 = 4,500 batches
```

**Calculation for WT-103:**
```
Dataset size: 539M characters
Tokens per batch: 64 * 128 = 8,192
Total tokens per epoch: 300 * 8,192 = 2.46M tokens
Coverage per epoch: 2.46M / 539M = 0.46%

After 15 epochs: 15 * 2.46M = 36.9M tokens seen (~7% of dataset)
```

**Why this matters:**
- Character-level models need to see more examples
- WT-103 is very large (539M characters)
- We're training on a subset for speed
- For production: train for 50+ epochs or use subword tokenization

### 7. Learning Rate Schedule

```python
lr = 5e-4              # Peak learning rate
warmup_iters = 100     # Gradually increase LR at start

# Warmup implementation
if global_step < warmup_iters:
    lr_scale = global_step / warmup_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_scale
```

**Why warmup?**
- Prevents large updates early when model is random
- More stable training
- Common practice for transformers

**Learning Rate Choice:**
- Too high: Training diverges
- Too low: Learns too slowly
- `5e-4` is a good default for AdamW

### 8. Memory Optimization: NumPy Stack

```python
# Before (slow, warning message)
x = [data[i:i+block_size] for i in ix]
x = torch.tensor(x, device=device)

# After (fast, no warning)
x = np.stack([data[i:i+block_size] for i in ix])
x = torch.from_numpy(x).to(device)
```

**Why?**
- Creating tensors from lists of arrays is slow
- Stack into single NumPy array first
- Then convert to PyTorch tensor
- ~2x faster data loading

---

## Training Process

### 1. Data Preparation

```python
# Load and combine train + validation for tokenizer
train_text = open(TRAIN_PATH, "r", encoding="utf-8").read()
valid_text = open(VALID_PATH, "r", encoding="utf-8").read()

# Build vocabulary from both to avoid KeyErrors
combined_text = train_text + valid_text
tok = CharTokenizer(combined_text)

# Encode to integers
train_data = np.array(tok.encode(train_text), dtype=np.int32)
val_data = np.array(tok.encode(valid_text), dtype=np.int32)
```

**Key Decision:** Build tokenizer from combined vocabulary
- Training set: 539M chars
- Validation set: 1.1M chars
- Combined vocab: 5006 unique characters (includes Unicode)

### 2. Training Loop

```python
for epoch in range(epochs):
    for it in range(iters_per_epoch):
        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(grad_accum_steps):
            # Get batch
            xb, yb = get_batch(train_data, batch_size, block_size, DEVICE)
            
            # Forward pass
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            accum_loss += loss.item()
        
        # Clip gradients (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Evaluate periodically
        if it % 50 == 0:
            losses = estimate_loss(model, train_data, val_data, ...)
            print(f"train: {losses['train']:.4f}, val: {losses['val']:.4f}")
```

### 3. Loss Function

We use **cross-entropy loss** which measures how well the model predicts the next token:

```python
def forward(self, idx, targets=None):
    # ... compute logits ...
    
    if targets is None:
        return logits
    
    # Cross-entropy: -log(P(correct token))
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (batch*seq, vocab_size)
        targets.view(-1)                    # (batch*seq,)
    )
    return logits, loss
```

**What's happening:**
- Model outputs probability distribution over vocabulary
- Loss is high if model assigns low probability to correct next character
- Training minimizes this loss → model learns to predict better

### 4. Text Generation

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=0.8):
    for _ in range(max_new_tokens):
        # Get predictions for last block_size tokens
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        
        # Focus on last position (next token prediction)
        logits = logits[:, -1, :] / temperature
        
        # Sample from probability distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat([idx, next_id], dim=1)
    
    return idx
```

**Temperature parameter:**
- `temperature = 0.1`: Nearly deterministic (picks highest probability)
- `temperature = 1.0`: Standard sampling from distribution
- `temperature = 2.0`: More random/creative

---

## Key Learnings

### 1. Dataset Size Matters

**Character-level tokenization** requires seeing much more data than subword tokenization:
- WT-103: 539M characters → only seeing ~7% in our training
- TinyShakespeare: 1M characters → can train on full dataset multiple times

**Lesson:** For production, use BPE or SentencePiece tokenization.

### 2. Model Size vs Training Time

Bigger isn't always better:
- 200K params: Too small, can't learn complex patterns
- 5M params: ✅ Sweet spot for M4
- 24M params: Too slow, impractical for iteration

**Lesson:** Start small, scale up only if needed.

### 3. The O(n²) Attention Cost

Doubling context length (128→256) **quadruples** attention computation.

**Solutions:**
- Keep context reasonable (128-256 for character-level)
- Use efficient attention (Flash Attention, linear attention)
- Consider longer context only for subword tokenization

### 4. Hardware-Specific Optimization

MPS on M4 is fast but:
- First iteration compiles kernels (appears frozen)
- Batch size sweet spot: 64-128
- Gradient accumulation helps simulate larger batches

**Lesson:** Profile and tune for your specific hardware.

### 5. Tokenization Strategy

Character-level is educational but:
- ✅ Simple to understand
- ✅ No vocabulary issues
- ❌ Very long sequences
- ❌ Hard to learn word-level semantics

**Better approach:** BPE with 50K vocabulary
- Shorter sequences (4x-10x compression)
- Learns subword patterns
- Industry standard

### 6. Training Convergence

Good quality requires:
- **Seeing enough tokens**: At least 10-100x model parameters
- **Enough iterations**: Until validation loss plateaus
- **Good hyperparameters**: LR, batch size, model capacity

For our 5M param model:
- Need ~50-500M tokens
- WT-103 has 539M, but we only see ~37M in 15 epochs
- More epochs would improve quality

---

## Next Steps to Improve

1. **Better Tokenization**
   - Implement BPE (Byte-Pair Encoding)
   - Use `tokenizers` library from Hugging Face
   - Reduce vocab from 5006 → 2000-8000 subwords

2. **Longer Training**
   - Train for 50+ epochs on WT-103
   - Or train to convergence (val loss stops improving)
   - Save checkpoints regularly

3. **Learning Rate Schedule**
   - Add cosine decay after warmup
   - Try learning rate finder
   - Experiment with different optimizers (Lion, AdaFactor)

4. **Model Architecture**
   - Try rotary position embeddings (RoPE)
   - Experiment with different normalization (RMSNorm)
   - Add more layers if you have time/compute

5. **Data Quality**
   - Filter WT-103 to remove non-English text
   - Or use cleaner dataset (OpenWebText, C4)
   - Data quality > quantity

6. **Evaluation**
   - Calculate perplexity: `exp(loss)`
   - Use standard benchmarks
   - Compare against baseline models

7. **Production Optimizations**
   - Mixed precision training (FP16/BF16)
   - Flash Attention for 2-3x speedup
   - Model quantization for inference
   - ONNX export for deployment

---

## References

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

### Tutorials
- [Karpathy's MinGPT](https://github.com/karpathy/minGPT) - Clean implementation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Production-quality minimal GPT
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations

### Documentation
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Fast.ai Course](https://course.fast.ai/) - Practical deep learning

---

## Conclusion

You've built a **working GPT-style transformer** from scratch! This implementation includes:

✅ Multi-head causal self-attention  
✅ Transformer blocks with residual connections  
✅ Character-level tokenization  
✅ Training on WT-103 Wikipedia text  
✅ Optimized for Apple Silicon M4  
✅ Text generation from trained model  

**Key takeaways:**
- Attention allows models to focus on relevant context
- Transformers are surprisingly simple architecturally
- Performance tuning is critical for practical training
- Bigger models need more data and time

Keep experimenting! Try different datasets, architectures, and hyperparameters to deepen your understanding. 🚀
