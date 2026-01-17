"""
Bigram Language Model for character-level text generation.
Core models and training utilities.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


# ============== Data Loading & Preprocessing ==============

def load_text(filepath="input.txt"):
    """Load text data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def create_vocabulary(text):
    """Create vocabulary mappings from text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, vocab_size, stoi, itos


def create_encoder_decoder(stoi, itos):
    """Create encode and decode functions."""
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda s: ''.join([itos[c] for c in s])
    return encode, decode


def prepare_data(text, stoi, train_split=0.9):
    """Encode text and split into train/validation sets."""
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return data, train_data, val_data


# ============== Batch Generation ==============

def get_batch(split, train_data, val_data, batch_size=32, block_size=8):
    """Generate a batch of data for training or validation."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# ============== Bigram Language Model ==============

class BigramLanguageModel(nn.Module):
    """
    Simple bigram language model.
    Predicts the next character based only on the current character.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============== Embedding Language Model ==============

class EmbeddingLanguageModel(nn.Module):
    """
    Language model with token embeddings + positional embeddings.
    """

    def __init__(self, vocab_size, n_embd=32, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============== Single Head Attention ==============

class Head(nn.Module):
    """A single head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class SingleHeadAttentionModel(nn.Module):
    """
    Language model with a single attention head and residual connection.
    """

    def __init__(self, vocab_size, n_embd=32, block_size=8, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, n_embd, block_size, dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Residual connection
        x = x + self.sa_head(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============== Training Utilities ==============

def estimate_loss(model, train_data, val_data, batch_size=32, block_size=8, eval_iters=200):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, train_data, val_data, batch_size=32, block_size=8,
                max_iters=10000, eval_interval=1000, eval_iters=200, lr=1e-3):
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for step in range(max_iters):
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters)
            print(f"Step {step}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
            history.append(losses)

    return history


# ============== Interpretability Utilities ==============

def save_weights_snapshot(model):
    """Save a snapshot of current model weights."""
    with torch.no_grad():
        return model.token_embedding_table.weight.cpu().numpy().copy()


def generate_with_temperature(model, idx, max_new_tokens, temperature=1.0):
    """Generate tokens with temperature-controlled sampling."""
    idx = idx.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def sequence_probability(model, sequence, stoi):
    """Calculate the probability of a sequence under the model."""
    if len(sequence) < 2:
        raise ValueError("Sequence must have at least 2 characters")

    for char in sequence:
        if char not in stoi:
            raise ValueError(f"Character '{char}' not in vocabulary")

    log_prob = 0.0
    transition_probs = []

    with torch.no_grad():
        for i in range(len(sequence) - 1):
            from_char = sequence[i]
            to_char = sequence[i + 1]

            from_idx = stoi[from_char]
            to_idx = stoi[to_char]

            logits = model.token_embedding_table.weight[from_idx]
            probs = F.softmax(logits, dim=-1)
            prob = probs[to_idx].item()

            transition_probs.append({
                'from': from_char,
                'to': to_char,
                'probability': prob
            })

            log_prob += np.log(prob + 1e-10)

    total_prob = np.exp(log_prob)
    perplexity = np.exp(-log_prob / (len(sequence) - 1))

    return {
        'probability': total_prob,
        'log_probability': log_prob,
        'perplexity': perplexity,
        'transition_probs': transition_probs
    }


def print_sequence_analysis(model, sequence, stoi):
    """Pretty print the sequence probability analysis."""
    result = sequence_probability(model, sequence, stoi)

    print("=" * 60)
    print(f"SEQUENCE ANALYSIS: '{sequence}'")
    print("=" * 60)
    print(f"\nOverall Statistics:")
    print(f"  Total Probability: {result['probability']:.2e}")
    print(f"  Log Probability: {result['log_probability']:.4f}")
    print(f"  Perplexity: {result['perplexity']:.4f}")
    print(f"\nTransition Breakdown:")
    print("-" * 40)

    for trans in result['transition_probs']:
        from_repr = repr(trans['from'])[1:-1]
        to_repr = repr(trans['to'])[1:-1]
        prob = trans['probability']
        print(f"  '{from_repr}' -> '{to_repr}': {prob:.6f}")

    print("-" * 40)

    min_trans = min(result['transition_probs'], key=lambda x: x['probability'])
    print(f"\nWeakest transition: '{min_trans['from']}' -> '{min_trans['to']}' ({min_trans['probability']:.6f})")


# ============== Setup Function ==============

def setup_bigram_model(text_path="input.txt", seed=1337):
    """Convenience function to set up everything for training."""
    torch.manual_seed(seed)

    text = load_text(text_path)
    chars, vocab_size, stoi, itos = create_vocabulary(text)
    encode, decode = create_encoder_decoder(stoi, itos)
    data, train_data, val_data = prepare_data(text, stoi)

    model = BigramLanguageModel(vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        'text': text,
        'chars': chars,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'encode': encode,
        'decode': decode,
        'data': data,
        'train_data': train_data,
        'val_data': val_data,
        'model': model,
        'device': device
    }
