"""
Visualization functions for the Bigram Language Model lab.
All plotting code is centralized here to keep the notebook clean.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ============== Training Visualization ==============

def plot_training_history(history, eval_interval=1000):
    """Plot training and validation loss curves."""
    train_hist = [h["train"].numpy() if hasattr(h["train"], 'numpy') else h["train"] for h in history]
    val_hist = [h["val"].numpy() if hasattr(h["val"], 'numpy') else h["val"] for h in history]

    steps = np.arange(len(train_hist)) * eval_interval

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_hist, label="Training Loss", color="indianred", alpha=0.7)
    plt.scatter(steps, train_hist, color="indianred")
    plt.plot(steps, val_hist, label="Validation Loss", color="coral", alpha=0.7)
    plt.scatter(steps, val_hist, color="coral")
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.legend()
    plt.title("Training Progress")
    plt.show()


# ============== Bigram Model Visualization ==============

def compute_actual_bigram_freqs(text, stoi, vocab_size):
    """Compute actual bigram frequencies from the corpus."""
    counts = np.zeros((vocab_size, vocab_size))

    for i in range(len(text) - 1):
        if text[i] in stoi and text[i+1] in stoi:
            from_idx = stoi[text[i]]
            to_idx = stoi[text[i+1]]
            counts[from_idx, to_idx] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    freqs = counts / row_sums

    return freqs


def visualize_learned_vs_actual(model, text, stoi, vocab_size, chars):
    """Side-by-side visualization of learned vs actual bigram distributions."""
    actual_freqs = compute_actual_bigram_freqs(text, stoi, vocab_size)

    with torch.no_grad():
        logits = model.token_embedding_table.weight
        learned_probs = F.softmax(logits, dim=-1).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Create character labels (handle special chars for display)
    char_labels = [repr(c)[1:-1] for c in chars]

    # Determine tick positions (show every nth character to avoid crowding)
    n_chars = len(chars)
    step = max(1, n_chars // 20)  # Show ~20 labels
    tick_positions = np.arange(0, n_chars, step)
    tick_labels = [char_labels[i] for i in tick_positions]

    im1 = axes[0].imshow(actual_freqs, cmap="viridis")
    axes[0].set_title("Actual Bigram Frequencies\n(Ground Truth from Text)")
    axes[0].set_xlabel("Next Character")
    axes[0].set_ylabel("Current Character")
    axes[0].set_xticks(tick_positions)
    axes[0].set_yticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label="Probability")

    im2 = axes[1].imshow(learned_probs, cmap="viridis")
    axes[1].set_title("Learned Probabilities\n(Model's Predictions)")
    axes[1].set_xlabel("Next Character")
    axes[1].set_ylabel("Current Character")
    axes[1].set_xticks(tick_positions)
    axes[1].set_yticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label="Probability")

    diff = np.abs(learned_probs - actual_freqs)
    im3 = axes[2].imshow(diff, cmap="Reds")
    axes[2].set_title("Absolute Difference\n(Model Error)")
    axes[2].set_xlabel("Next Character")
    axes[2].set_ylabel("Current Character")
    axes[2].set_xticks(tick_positions)
    axes[2].set_yticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, label="Error")

    plt.tight_layout()
    plt.show()

    print(f"Mean absolute error: {diff.mean():.6f}")
    print(f"Max absolute error: {diff.max():.6f}")


def visualize_top_predictions(model, char, stoi, itos, k=15):
    """Visualize top predictions as a bar chart."""
    if char not in stoi:
        raise ValueError(f"Character '{char}' not in vocabulary")

    char_idx = stoi[char]

    with torch.no_grad():
        logits = model.token_embedding_table.weight[char_idx]
        probs = F.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, k)

    chars_list = [repr(itos[idx.item()])[1:-1] for idx in top_indices]
    probs_list = [p.item() for p in top_probs]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(len(chars_list)), probs_list, color='steelblue')
    plt.xticks(range(len(chars_list)), chars_list, fontsize=10)
    plt.xlabel("Next Character")
    plt.ylabel("Probability")
    plt.title(f"Top {k} Predictions After '{repr(char)[1:-1]}'")

    for bar, prob in zip(bars, probs_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def compare_embeddings_before_after(model_before_weights, model_after, chars):
    """Side-by-side comparison of embeddings before and after training."""
    with torch.no_grad():
        after_weights = model_after.token_embedding_table.weight.cpu().numpy()

    before_probs = F.softmax(torch.tensor(model_before_weights), dim=-1).numpy()
    after_probs = F.softmax(torch.tensor(after_weights), dim=-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Create character labels (handle special chars for display)
    char_labels = [repr(c)[1:-1] for c in chars]

    # Determine tick positions (show every nth character to avoid crowding)
    n_chars = len(chars)
    step = max(1, n_chars // 20)  # Show ~20 labels
    tick_positions = np.arange(0, n_chars, step)
    tick_labels = [char_labels[i] for i in tick_positions]

    im1 = axes[0].imshow(before_probs, cmap="viridis")
    axes[0].set_title("Before Training\n(Random Initialization)")
    axes[0].set_xlabel("Next Character")
    axes[0].set_ylabel("Current Character")
    axes[0].set_xticks(tick_positions)
    axes[0].set_yticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label="Probability")

    im2 = axes[1].imshow(after_probs, cmap="viridis")
    axes[1].set_title("After Training\n(Learned Probabilities)")
    axes[1].set_xlabel("Next Character")
    axes[1].set_ylabel("Current Character")
    axes[1].set_xticks(tick_positions)
    axes[1].set_yticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label="Probability")

    diff = after_probs - before_probs
    im3 = axes[2].imshow(diff, cmap="RdBu", vmin=-diff.max(), vmax=diff.max())
    axes[2].set_title("Change (After - Before)\n(Red=increased, Blue=decreased)")
    axes[2].set_xlabel("Next Character")
    axes[2].set_ylabel("Current Character")
    axes[2].set_xticks(tick_positions)
    axes[2].set_yticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, label="Probability Change")

    plt.tight_layout()
    plt.show()

    print(f"Total weight change (L2 norm): {np.linalg.norm(after_weights - model_before_weights):.4f}")


# ============== Embedding Model Visualization ==============

def plot_token_embedding_pca(model, chars):
    """Visualize token embeddings in 2D using PCA."""
    from sklearn.decomposition import PCA

    with torch.no_grad():
        embeddings = model.token_embedding_table.weight.cpu().numpy()

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    for i, char in enumerate(chars):
        x, y = embeddings_2d[i]

        if char.isalpha() and char.islower():
            color = 'steelblue'
        elif char.isalpha() and char.isupper():
            color = 'coral'
        elif char.isdigit():
            color = 'green'
        elif char in ' \n\t':
            color = 'red'
        else:
            color = 'gray'

        plt.scatter(x, y, c=color, s=100, alpha=0.7)
        label = repr(char)[1:-1]
        plt.annotate(label, (x, y), fontsize=8, ha='center', va='bottom')

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Token Embeddings (PCA)\nBlue=lowercase, Orange=uppercase, Red=whitespace, Gray=punctuation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_token_embedding_similarity(model, chars):
    """Visualize cosine similarity between all token embeddings."""
    with torch.no_grad():
        embeddings = model.token_embedding_table.weight.cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    similarity = normalized @ normalized.T

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(similarity, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xlabel("Character")
    ax.set_ylabel("Character")
    ax.set_title("Token Embedding Similarity\n(Red = similar, Blue = dissimilar)")

    step = max(1, len(chars) // 30)
    ax.set_xticks(np.arange(0, len(chars), step))
    ax.set_yticks(np.arange(0, len(chars), step))
    ax.set_xticklabels([repr(c)[1:-1] for c in chars[::step]], rotation=45, ha='right')
    ax.set_yticklabels([repr(c)[1:-1] for c in chars[::step]])

    plt.tight_layout()
    plt.show()


def find_similar_characters(model, char, chars, stoi, top_k=10):
    """Find the most similar characters based on embedding similarity."""
    if char not in stoi:
        raise ValueError(f"Character '{char}' not in vocabulary")

    with torch.no_grad():
        embeddings = model.token_embedding_table.weight.cpu().numpy()

    char_idx = stoi[char]
    char_emb = embeddings[char_idx]

    norms = np.linalg.norm(embeddings, axis=1)
    char_norm = np.linalg.norm(char_emb)
    similarities = (embeddings @ char_emb) / (norms * char_norm + 1e-8)

    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        if idx != char_idx:
            results.append((chars[idx], similarities[idx]))
        if len(results) >= top_k:
            break

    return results


# ============== Attention Visualization ==============

def show_attention(model, text, encode, decode):
    """Visualize attention weights for a given input text."""
    tokens = encode(text)[:model.block_size]
    x = torch.tensor([tokens])
    T = len(tokens)

    # Get attention weights
    tok_emb = model.token_embedding_table(x)
    pos_emb = model.position_embedding_table(torch.arange(T))
    x_emb = tok_emb + pos_emb

    head = model.sa_head
    k, q = head.key(x_emb), head.query(x_emb)

    # Compute attention with causal mask
    scores = q @ k.transpose(-2, -1) * head.head_size**-0.5
    mask = torch.tril(torch.ones(T, T))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    wei = F.softmax(scores, dim=-1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(wei[0].detach().numpy(), cmap='Blues')
    labels = [repr(decode([t]))[1:-1] for t in tokens]
    plt.xticks(range(T), labels)
    plt.yticks(range(T), labels)
    plt.xlabel('Attending to')
    plt.ylabel('Attending from')
    plt.title(f'Attention Pattern: "{text}"')
    plt.colorbar(label='Weight')
    plt.tight_layout()
    plt.show()


def compare_models(history_list, labels, eval_interval=1000):
    """Plot validation loss curves for multiple models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['gray', 'steelblue', 'coral', 'green', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (history, label) in enumerate(zip(history_list, labels)):
        val_losses = [h['val'].item() if hasattr(h['val'], 'item') else h['val'] for h in history]
        steps = np.arange(len(val_losses)) * eval_interval
        ax.plot(steps, val_losses, f'{markers[i % len(markers)]}-',
                label=label, color=colors[i % len(colors)], alpha=0.7)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============== Linear Representation Hypothesis ==============

def explore_capitalization_direction(model, letter, stoi):
    """
    Explore the capitalization direction for a single letter.
    Shows embeddings, difference vector, and angle with a reference direction.
    """
    # Get embeddings
    lower_emb = model.token_embedding_table.weight[stoi[letter]].detach().numpy()
    upper_emb = model.token_embedding_table.weight[stoi[letter.upper()]].detach().numpy()
    diff = upper_emb - lower_emb

    # Compute reference direction (A - a) for comparison
    ref_lower = model.token_embedding_table.weight[stoi['a']].detach().numpy()
    ref_upper = model.token_embedding_table.weight[stoi['A']].detach().numpy()
    ref_diff = ref_upper - ref_lower

    # Compute angle between this letter's direction and reference
    cos_sim = np.dot(diff, ref_diff) / (np.linalg.norm(diff) * np.linalg.norm(ref_diff) + 1e-8)
    angle_degrees = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi

    print(f"Capitalization direction for '{letter}' → '{letter.upper()}'")
    print("=" * 50)
    print(f"\nEmbedding for '{letter}':       {lower_emb[:6].round(2)}...")
    print(f"Embedding for '{letter.upper()}':       {upper_emb[:6].round(2)}...")
    print(f"Difference ({letter.upper()} - {letter}):    {diff[:6].round(2)}...")
    print(f"\n📐 Angle from reference (A-a):  {angle_degrees:.1f}°")
    print(f"   Cosine similarity:           {cos_sim:.3f}")
    print(f"\nInterpretation:")
    if angle_degrees < 30:
        print(f"   ✓ Very similar to (A-a) direction! The model learned a consistent pattern.")
    elif angle_degrees < 60:
        print(f"   ~ Somewhat similar to (A-a) direction.")
    else:
        print(f"   ✗ Different from (A-a) direction. This letter has its own pattern.")


def plot_capitalization_similarity(model, stoi):
    """
    Plot cosine similarity matrix between all capitalization directions.
    Shows whether the model learned a consistent capitalization direction.
    """
    # Letter pairs to analyze
    letter_pairs = [('a','A'), ('b','B'), ('c','C'), ('d','D'), ('e','E'), ('f','F'),
                    ('h','H'), ('i','I'), ('l','L'), ('m','M'), ('n','N'), ('o','O'),
                    ('r','R'), ('s','S'), ('t','T'), ('w','W')]

    directions = []
    labels = []

    for lower, upper in letter_pairs:
        lower_emb = model.token_embedding_table.weight[stoi[lower]].detach().numpy()
        upper_emb = model.token_embedding_table.weight[stoi[upper]].detach().numpy()
        directions.append(upper_emb - lower_emb)
        labels.append(f"{upper}-{lower}")

    directions = np.array(directions)

    # Compute cosine similarity between all pairs
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized = directions / (norms + 1e-8)
    similarities = normalized @ normalized.T

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(similarities, cmap='RdBu', vmin=-1, vmax=1)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.colorbar(label='Cosine Similarity')
    plt.title('How Similar Are the Capitalization Directions?\n(Red = similar, Blue = opposite)')
    plt.tight_layout()
    plt.show()

    # Summary statistics
    upper_triangle = similarities[np.triu_indices(len(labels), k=1)]
    avg_sim = upper_triangle.mean()

    print(f"\nAverage similarity: {avg_sim:.3f}")
    print("\nInterpretation:")
    if avg_sim > 0.5:
        print("  ✓ Mostly red → The model learned ONE consistent capitalization direction!")
    elif avg_sim > 0.2:
        print("  ~ Mixed colors → The model partially learned capitalization.")
    else:
        print("  ✗ Mixed/blue → Each letter has its own unique pattern.")
