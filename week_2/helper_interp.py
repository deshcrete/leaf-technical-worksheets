"""
Mechanistic Interpretability Helper Functions

This module provides tools for probing neural network activations to understand
what concepts the network has learned at different layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# MODEL CREATION AND TRAINING
# =============================================================================

def create_model():
    """Create the MNIST classification model."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )


def create_optimizer(model, lr=0.001):
    """Create an Adam optimizer for the model."""
    return optim.Adam(model.parameters(), lr=lr)


def train_model(model, optimizer, train_loader, test_loader, epochs=20, verbose=True):
    """
    Train the MNIST model.

    Args:
        model: The neural network model
        optimizer: The optimizer to use
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        verbose: Whether to print progress

    Returns:
        Dictionary with training history (train_loss, test_loss, test_accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        test_loss /= len(test_loader)
        accuracy = correct / total

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(accuracy)

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    return history


def load_data(train_csv, test_csv):
    """
    Load MNIST data from CSV files.

    Args:
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_labels = torch.tensor(train_df.iloc[:, 0].values)
    train_images = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32) / 255.0

    test_labels = torch.tensor(test_df.iloc[:, 0].values)
    test_images = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32) / 255.0

    return train_images, train_labels, test_images, test_labels


def setup_data(batch_size=64):
    """
    Download MNIST data (if needed) and return images, labels, and dataloaders.

    This function handles everything needed to get the data ready:
    1. Downloads MNIST using torchvision (if CSV files don't exist)
    2. Converts to CSV format for easy inspection
    3. Loads the data as tensors
    4. Creates DataLoaders for training

    Args:
        batch_size: Batch size for the dataloaders (default: 64)

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels, train_loader, test_loader)
    """
    import os

    # Download and convert to CSV if needed
    if not os.path.exists('mnist_train_small.csv') or not os.path.exists('mnist_test.csv'):
        print("Downloading MNIST dataset...")
        from torchvision import datasets

        train_data = datasets.MNIST(root='./data', train=True, download=True)
        test_data = datasets.MNIST(root='./data', train=False, download=True)

        print("Converting to CSV format...")

        # Training data
        train_imgs = train_data.data.numpy().reshape(-1, 784)
        train_lbls = train_data.targets.numpy()
        train_df = pd.DataFrame(train_imgs)
        train_df.insert(0, 'label', train_lbls)
        train_df.to_csv('mnist_train_small.csv', index=False)

        # Test data
        test_imgs = test_data.data.numpy().reshape(-1, 784)
        test_lbls = test_data.targets.numpy()
        test_df = pd.DataFrame(test_imgs)
        test_df.insert(0, 'label', test_lbls)
        test_df.to_csv('mnist_test.csv', index=False)

        print("Download complete!")

    # Load the data
    train_images, train_labels, test_images, test_labels = load_data(
        'mnist_train_small.csv',
        'mnist_test.csv'
    )

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_images, train_labels, test_images, test_labels, batch_size=batch_size
    )

    print(f"Loaded {len(train_images)} training samples and {len(test_images)} test samples")

    return train_images, train_labels, test_images, test_labels, train_loader, test_loader


def create_dataloaders(train_images, train_labels, test_images, test_labels, batch_size=64):
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def plot_training_history(history):
    """Plot training loss, test loss, and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["test_loss"], c="tomato", label="Test Loss")
    plt.plot(epochs, history["train_loss"], c="indigo", label="Train Loss")
    plt.plot(epochs, history["test_accuracy"], c="skyblue", label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.title("Training Progress for MNIST NN", fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def visualize_network_architecture():
    """Visualize the neural network architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))

    layers = [
        {'name': 'Input\n(Flattened)', 'neurons': 784, 'display': 8, 'color': '#E8F4FD', 'x': 0.5},
        {'name': 'Hidden 1\n+ ReLU', 'neurons': 32, 'display': 8, 'color': '#B8D4E8', 'x': 2.5},
        {'name': 'Hidden 2\n+ ReLU', 'neurons': 16, 'display': 8, 'color': '#88B4D8', 'x': 4.5},
        {'name': 'Output\n(Logits)', 'neurons': 10, 'display': 10, 'color': '#5894C8', 'x': 6.5}
    ]

    for layer in layers:
        x = layer['x']
        n_display = layer['display']
        y_positions = np.linspace(1, 7, n_display)

        for y in y_positions:
            circle = plt.Circle((x, y), 0.25, color=layer['color'], ec='#2C5282', linewidth=2)
            ax.add_patch(circle)

        if layer['neurons'] > n_display:
            ax.text(x, 0.3, '...', fontsize=14, ha='center', va='center', fontweight='bold')
            ax.text(x, -0.3, f'({layer["neurons"]} total)', fontsize=10, ha='center', va='center')

        ax.text(x, 8.2, layer['name'], fontsize=12, ha='center', va='center', fontweight='bold')

    # Draw connections
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + 0.25
        x2 = layers[i+1]['x'] - 0.25
        y1_positions = np.linspace(1, 7, layers[i]['display'])
        y2_positions = np.linspace(1, 7, layers[i+1]['display'])

        for y1 in y1_positions[::2]:
            for y2 in y2_positions[::2]:
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.15, linewidth=0.5)

    # Add arrows
    for start, end in [(1.2, 1.8), (3.2, 3.8), (5.2, 5.8)]:
        ax.annotate('', xy=(end, 4), xytext=(start, 4),
                    arrowprops=dict(arrowstyle='->', color='#2C5282', lw=2))

    ax.text(1.5, 4.5, '784->32', fontsize=9, ha='center', style='italic')
    ax.text(3.5, 4.5, '32->16', fontsize=9, ha='center', style='italic')
    ax.text(5.5, 4.5, '16->10', fontsize=9, ha='center', style='italic')

    ax.set_title('MNIST Neural Network Architecture\n784 -> 32 -> 16 -> 10',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_sample_digits(images, labels):
    """Display one sample of each digit 0-9."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    samples = {}
    for img, lbl in zip(images, labels):
        digit = lbl.item()
        if digit not in samples:
            samples[digit] = img
        if len(samples) == 10:
            break

    for digit in range(10):
        ax = axes[digit // 5, digit % 5]
        img = samples[digit].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Digit: {digit}', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Sample MNIST Digits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_prediction(model, image, true_label):
    """Show input digit, raw logits, softmax transformation, and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze()

    predicted = probs.argmax().item()
    confidence = probs[predicted].item()
    logits_np = logits.squeeze().numpy()
    probs_np = probs.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Input image
    axes[0].imshow(image.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Input (Label: {true_label})', fontweight='bold')
    axes[0].axis('off')

    # 2. Raw logits
    colors = ['#2ecc71' if i == predicted else '#3498db' for i in range(10)]
    axes[1].bar(range(10), logits_np, color=colors)
    axes[1].set_xlabel('Digit')
    axes[1].set_ylabel('Logit value')
    axes[1].set_title('1. Raw Logits', fontweight='bold')
    axes[1].set_xticks(range(10))
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 3. After exponentiation (e^logit) - intermediate step
    exp_logits = np.exp(logits_np)
    colors = ['#e74c3c' if i == predicted else '#f39c12' for i in range(10)]
    axes[2].bar(range(10), exp_logits, color=colors)
    axes[2].set_xlabel('Digit')
    axes[2].set_ylabel('e^logit')
    axes[2].set_title(f'2. After e^x (sum={exp_logits.sum():.1f})', fontweight='bold')
    axes[2].set_xticks(range(10))

    # 4. Probability distribution (after normalizing)
    colors = ['#27ae60' if i == predicted else '#95a5a6' for i in range(10)]
    bars = axes[3].bar(range(10), probs_np * 100, color=colors)
    axes[3].set_xlabel('Digit')
    axes[3].set_ylabel('Probability (%)')
    axes[3].set_title(f'3. After Softmax → Prediction: {predicted}', fontweight='bold')
    axes[3].set_xticks(range(10))
    axes[3].set_ylim(0, 105)

    for i, (bar, p) in enumerate(zip(bars, probs_np)):
        if p > 0.05:
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{p*100:.1f}%', ha='center', fontsize=8)

    plt.suptitle(f'Logits → Softmax → Probability (Confidence: {confidence*100:.1f}%)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ACTIVATION EXTRACTION AND VISUALIZATION
# =============================================================================

def get_activations(model, images):
    """
    Extract activations from each layer of the model.

    Args:
        model: PyTorch Sequential model
        images: Input tensor of images

    Returns:
        List of activation tensors from each Linear and ReLU layer
    """
    activations = []
    x = images

    for layer in model:
        x = layer(x)
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Linear):
            activations.append(x.detach())

    return activations


def plot_activations(model, image, label=None):
    """
    Visualize activations at each layer for a single input image.

    Args:
        model: The trained MNIST model
        image: Single image tensor (784,) or (1, 784)
        label: Optional true label for the image
    """
    model.eval()

    # Ensure image has correct shape
    if image.dim() == 1:
        image = image.unsqueeze(0)

    acts = get_activations(model, image)

    # Get prediction
    with torch.no_grad():
        probs = F.softmax(acts[-1], dim=1).squeeze()
    predicted = probs.argmax().item()
    confidence = probs[predicted].item()

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # Input image
    ax_img = fig.add_subplot(2, 4, 1)
    ax_img.imshow(image.squeeze().reshape(28, 28), cmap='gray')
    title = f'Input'
    if label is not None:
        title += f' (True: {label})'
    ax_img.set_title(title, fontweight='bold')
    ax_img.axis('off')

    # Layer names
    layer_info = [
        ('Layer 0: Linear 1', 32, 'steelblue'),
        ('Layer 1: ReLU 1', 32, 'steelblue'),
        ('Layer 2: Linear 2', 16, 'coral'),
        ('Layer 3: ReLU 2', 16, 'coral'),
        ('Layer 4: Output', 10, 'seagreen'),
    ]

    # Plot activations for each layer
    positions = [(2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 4, 6), (2, 4, 7)]

    for idx, ((name, size, color), pos) in enumerate(zip(layer_info, positions)):
        ax = fig.add_subplot(*pos)
        act_values = acts[idx].squeeze().numpy()

        if idx == 4:  # Output layer - special coloring
            colors = ['#27ae60' if i == predicted else '#95a5a6' for i in range(10)]
            ax.bar(range(len(act_values)), act_values, color=colors)
            ax.set_xticks(range(10))
        else:
            ax.bar(range(len(act_values)), act_values, color=color, alpha=0.7)

        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xlabel('Neuron')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add prediction info
    fig.suptitle(f'Activations Through the Network → Prediction: {predicted} ({confidence:.1%})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def get_sample_by_digit(images, labels, digit):
    """
    Get a random sample image for a specific digit.

    Args:
        images: Tensor of images
        labels: Tensor of labels
        digit: The digit to find (0-9)

    Returns:
        Tuple of (image, label) or (None, None) if not found
    """
    # Find all indices where the label matches the digit
    matching_indices = (labels == digit).nonzero(as_tuple=True)[0]

    if len(matching_indices) == 0:
        return None, None

    # Randomly select one using torch (which uses different random state than numpy)
    rand_idx = torch.randint(len(matching_indices), (1,)).item()
    idx = matching_indices[rand_idx].item()
    return images[idx], labels[idx].item()


def get_sample_by_index(images, labels, index):
    """
    Get a sample image by index.

    Args:
        images: Tensor of images
        labels: Tensor of labels
        index: Index of the sample to retrieve

    Returns:
        Tuple of (image, label)
    """
    return images[index], labels[index].item()


# =============================================================================
# PROBE FUNCTIONS
# =============================================================================

def _collect_probe_data(model, data_loader, layer_num, positive_digits, negative_digits):
    """
    Internal function to collect activations and labels for probe training.

    Args:
        model: The trained MNIST model
        data_loader: DataLoader for the dataset
        layer_num: Which layer's activations to extract (0-4)
        positive_digits: List of digits to label as 1
        negative_digits: List of digits to label as 0

    Returns:
        Tuple of (activations tensor, labels tensor)
    """
    model.eval()
    X_data = []
    y_data = []

    with torch.no_grad():
        for images, labels in data_loader:
            activations = get_activations(model, images)
            hidden_layer = activations[layer_num]

            for i, label in enumerate(labels):
                digit = label.item()
                if digit in positive_digits:
                    X_data.append(hidden_layer[i])
                    y_data.append(1)
                elif digit in negative_digits:
                    X_data.append(hidden_layer[i])
                    y_data.append(0)

    return torch.stack(X_data), torch.tensor(y_data, dtype=torch.float32)


def train_probe(model, train_loader, test_loader, positive_digits, negative_digits,
                layer_num, epochs=50, learning_rate=0.01, verbose=True):
    """
    Train a linear probe to classify between two groups of digits based on activations.

    A probe is a simple linear classifier trained on the activations of a hidden layer.
    If the probe achieves high accuracy, it suggests that the network has learned to
    represent the distinction between the digit groups at that layer.

    Args:
        model: The trained MNIST model to probe
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        positive_digits: List of digits to classify as positive (1)
                        e.g., [0, 6, 8, 9] for "digits with loops"
        negative_digits: List of digits to classify as negative (0)
                        e.g., [1, 2, 3, 4, 5, 7] for "digits without loops"
        layer_num: Which layer to probe (0-4)
                   0 = after first Linear (32 neurons)
                   1 = after first ReLU (32 neurons)
                   2 = after second Linear (16 neurons)
                   3 = after second ReLU (16 neurons)
                   4 = output layer (10 neurons)
        epochs: Number of training epochs for the probe
        learning_rate: Learning rate for probe training
        verbose: Whether to print accuracy

    Returns:
        Tuple of (trained probe, test accuracy)
    """
    # Determine layer size based on layer number
    layer_sizes = {0: 32, 1: 32, 2: 16, 3: 16, 4: 10}
    layer_size = layer_sizes[layer_num]

    # Collect training data
    train_X, train_y = _collect_probe_data(
        model, train_loader, layer_num, positive_digits, negative_digits
    )

    # Create and train probe
    probe = nn.Linear(layer_size, 1)
    probe_optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        probe_optimizer.zero_grad()
        outputs = probe(train_X).squeeze()
        loss = criterion(outputs, train_y)
        loss.backward()
        probe_optimizer.step()

    # Evaluate probe
    probe.eval()
    test_X, test_y = _collect_probe_data(
        model, test_loader, layer_num, positive_digits, negative_digits
    )

    with torch.no_grad():
        predictions = (torch.sigmoid(probe(test_X).squeeze()) > 0.5).long()
        accuracy = (predictions == test_y.long()).float().mean().item()

    if verbose:
        print(f'Probe accuracy: {accuracy:.4f}')

    return probe, accuracy


def visualize_probe_weights(probe, title="Probe Weights"):
    """
    Visualize the learned weights of a probe.

    The weights show which neurons the probe relies on most heavily
    to make its classification decision.

    Args:
        probe: Trained linear probe
        title: Title for the plot
    """
    plt.figure(figsize=(12, 2))
    weights = probe.weight.detach().numpy()
    plt.imshow(weights, cmap="plasma", aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Neuron Index')
    plt.title(title, fontweight='bold')
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def compare_probes_across_layers(model, train_loader, test_loader,
                                  positive_digits, negative_digits,
                                  layers_to_probe=[0, 1, 2, 3, 4]):
    """
    Train probes at multiple layers and compare their accuracies.

    This helps understand at which layer the network learns to distinguish
    between the two groups of digits.

    Args:
        model: The trained MNIST model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        positive_digits: Digits to classify as positive
        negative_digits: Digits to classify as negative
        layers_to_probe: List of layer indices to probe

    Returns:
        Dictionary mapping layer number to (probe, accuracy)
    """
    results = {}
    layer_names = {
        0: "Linear 1 (32)",
        1: "ReLU 1 (32)",
        2: "Linear 2 (16)",
        3: "ReLU 2 (16)",
        4: "Output (10)"
    }

    print(f"Probing: {positive_digits} vs {negative_digits}\n")

    for layer_num in layers_to_probe:
        print(f"Layer {layer_num} ({layer_names[layer_num]}): ", end="")
        probe, acc = train_probe(
            model, train_loader, test_loader,
            positive_digits, negative_digits,
            layer_num, verbose=True
        )
        results[layer_num] = (probe, acc)

    # Plot comparison
    layers = list(results.keys())
    accuracies = [results[l][1] for l in layers]

    plt.figure(figsize=(10, 5))
    bars = plt.bar([layer_names[l] for l in layers], accuracies, color='steelblue')
    plt.ylabel('Probe Accuracy')
    plt.title(f'Probe Accuracy by Layer\n{positive_digits} vs {negative_digits}', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2%}', ha='center', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


def probe_all_digit_pairs(model, train_loader, test_loader, layer_num=2):
    """
    Train probes to distinguish between all pairs of digits.

    This reveals which digits the network finds easy or hard to distinguish
    at a given layer.

    Args:
        model: The trained MNIST model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        layer_num: Which layer to probe

    Returns:
        Dictionary mapping digit pairs to accuracy
    """
    results = {}

    for i in range(10):
        for j in range(i + 1, 10):
            probe, acc = train_probe(
                model, train_loader, test_loader,
                [i], [j], layer_num, verbose=False
            )
            results[(i, j)] = acc
            print(f"({i}, {j}): {acc:.4f}")

    # Create heatmap
    heatmap = np.ones((10, 10)) * np.nan
    for (i, j), acc in results.items():
        heatmap[i, j] = acc
        heatmap[j, i] = acc  # Symmetric

    plt.figure(figsize=(8, 7))
    plt.imshow(heatmap, cmap='RdYlGn', vmin=0.9, vmax=1.0)
    plt.colorbar(label='Probe Accuracy')
    plt.xlabel('Digit')
    plt.ylabel('Digit')
    plt.title(f'Pairwise Digit Discrimination (Layer {layer_num})', fontweight='bold')
    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(10):
        for j in range(10):
            if not np.isnan(heatmap[i, j]):
                color = 'white' if heatmap[i, j] < 0.95 else 'black'
                plt.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color=color)

    plt.tight_layout()
    plt.show()

    return results
