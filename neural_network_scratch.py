"""
╔══════════════════════════════════════════════════════════════════╗
║           NEURAL NETWORK — BUILT FROM SCRATCH                   ║
║           No PyTorch. No TensorFlow. Pure NumPy.                ║
║           Author : Ammar                                         ║
║           GitHub : github.com/ammar                              ║
╚══════════════════════════════════════════════════════════════════╝

What this does:
  - Implements a fully connected neural network using only NumPy
  - Trains on the XOR problem (classic AI benchmark)
  - Shows loss decreasing in real-time during training
  - Visualizes decision boundary with ASCII art
  - Demonstrates forward pass, backprop, and gradient descent manually
"""

import numpy as np
import time

# ─────────────────────────────────────────────
#  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def sigmoid(x):
    """Squashes any value into range (0, 1)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Gradient of sigmoid — needed for backpropagation"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """Rectified Linear Unit — fast and powerful"""
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ─────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────

def binary_cross_entropy(y_true, y_pred):
    """Standard loss for binary classification"""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ─────────────────────────────────────────────
#  NEURAL NETWORK CLASS
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    A fully connected feedforward neural network.
    Supports any number of hidden layers.
    Uses Sigmoid activations + manual backpropagation.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, seed=42):
        """
        Args:
            layer_sizes : list of ints, e.g. [2, 8, 8, 1]
                          means: 2 inputs → 8 → 8 → 1 output
            learning_rate : step size for gradient descent
        """
        np.random.seed(seed)
        self.lr     = learning_rate
        self.layers = layer_sizes
        self.depth  = len(layer_sizes) - 1

        # ── He Initialization (better than random)
        self.weights = []
        self.biases  = []

        for i in range(self.depth):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He init: good for relu; scaled for sigmoid
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        self.loss_history = []

    # ── FORWARD PASS ─────────────────────────

    def forward(self, X):
        """
        Pass input through all layers.
        Stores intermediate values needed for backprop.
        """
        self.activations = [X]   # a[0] = input
        self.z_values    = []    # pre-activation values

        current = X
        for i in range(self.depth):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            # Last layer: sigmoid (output probability)
            # Hidden layers: sigmoid too (can swap to relu)
            a = sigmoid(z)
            self.activations.append(a)
            current = a

        return current  # final prediction

    # ── BACKWARD PASS (Backpropagation) ──────

    def backward(self, X, y_true):
        """
        Compute gradients using chain rule.
        Update weights and biases.
        """
        m       = X.shape[0]   # number of samples
        y_pred  = self.activations[-1]

        # Output layer error
        delta = (y_pred - y_true) * sigmoid_derivative(self.z_values[-1])

        grad_W = [None] * self.depth
        grad_b = [None] * self.depth

        # Propagate error backwards
        for i in reversed(range(self.depth)):
            a_prev       = self.activations[i]
            grad_W[i]    = (a_prev.T @ delta) / m
            grad_b[i]    = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * sigmoid_derivative(self.z_values[i - 1])

        # ── Gradient Descent Update
        for i in range(self.depth):
            self.weights[i] -= self.lr * grad_W[i]
            self.biases[i]  -= self.lr * grad_b[i]

    # ── TRAINING LOOP ────────────────────────

    def train(self, X, y, epochs=5000, verbose=True):
        """
        Run forward + backward pass for N epochs.
        """
        print("\n" + "═" * 58)
        print("  TRAINING STARTED")
        print("  Architecture :", " → ".join(str(s) for s in self.layers))
        print("  Learning Rate:", self.lr)
        print("  Epochs       :", epochs)
        print("═" * 58)

        bar_width = 30

        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss   = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            self.backward(X, y)

            if verbose and (epoch % 500 == 0 or epoch == 1):
                progress   = epoch / epochs
                filled     = int(bar_width * progress)
                bar        = "█" * filled + "░" * (bar_width - filled)
                accuracy   = self.accuracy(X, y) * 100
                print(f"  Epoch {epoch:5d}/{epochs} │{bar}│ Loss: {loss:.6f} │ Acc: {accuracy:.1f}%")

        print("═" * 58)
        print(f"  ✅ TRAINING COMPLETE — Final Loss: {self.loss_history[-1]:.6f}")
        print("═" * 58 + "\n")

    # ── PREDICTION & METRICS ─────────────────

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    # ── ASCII DECISION BOUNDARY ──────────────

    def visualize_decision_boundary(self, title="Decision Boundary"):
        """
        Renders an ASCII heatmap of what the network has learned.
        Works in any terminal — no matplotlib needed.
        """
        print(f"\n  {title}")
        print("  " + "─" * 42)

        cols, rows = 40, 20
        chars = " ░▒▓█"

        for row in range(rows):
            line = "  │"
            for col in range(cols):
                x1 = col / (cols - 1)
                x2 = 1 - row / (rows - 1)
                val = self.forward(np.array([[x1, x2]]))[0, 0]
                idx = int(val * (len(chars) - 1))
                line += chars[idx]
            print(line + "│")

        print("  " + "─" * 42)
        print("  Legend: ' ' = class 0 (low)   '█' = class 1 (high)\n")

    # ── LOSS CURVE (ASCII) ───────────────────

    def plot_loss_ascii(self):
        """Prints loss curve as ASCII chart"""
        print("  LOSS CURVE (training progress)")
        print("  " + "─" * 52)

        n       = len(self.loss_history)
        samples = 50
        step    = max(1, n // samples)
        values  = self.loss_history[::step]
        height  = 12

        max_v = max(values)
        min_v = min(values)
        rng   = max_v - min_v or 1e-9

        for row in range(height, -1, -1):
            threshold = min_v + (row / height) * rng
            line = f"  {threshold:.3f} │"
            for v in values:
                line += "▄" if v >= threshold else " "
            print(line)

        print("         └" + "─" * len(values))
        print(f"          0{'Epochs':>46}{n}\n")


# ─────────────────────────────────────────────
#  DEMO: XOR PROBLEM
# ─────────────────────────────────────────────

def demo_xor():
    """
    XOR is the classic test for neural networks.
    It CANNOT be solved by a single layer (not linearly separable).
    A proper multi-layer network solves it perfectly.
    """
    print("\n" + "╔" + "═" * 56 + "╗")
    print("║  DEMO 1: XOR Problem — The Classic AI Benchmark         ║")
    print("╚" + "═" * 56 + "╝")
    print("""
  XOR Truth Table:
  ┌───────┬───────┬────────┐
  │  x₁   │  x₂   │  y     │
  ├───────┼───────┼────────┤
  │   0   │   0   │   0    │
  │   0   │   1   │   1    │
  │   1   │   0   │   1    │
  │   1   │   1   │   0    │
  └───────┴───────┴────────┘
  A single-layer network CANNOT learn XOR.
  A 2-layer network CAN. Watch it happen below.
""")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0],    [1],    [1],    [0]])

    # Architecture: 2 inputs → 8 hidden → 8 hidden → 1 output
    nn = NeuralNetwork(layer_sizes=[2, 8, 8, 1], learning_rate=0.5)
    nn.train(X, y, epochs=5000)

    # Results
    print("  PREDICTIONS AFTER TRAINING:")
    print("  ┌───────┬───────┬──────────┬──────────┬─────────┐")
    print("  │  x₁   │  x₂   │  Target  │  Output  │  Match  │")
    print("  ├───────┼───────┼──────────┼──────────┼─────────┤")

    y_pred_raw = nn.forward(X)
    y_pred     = nn.predict(X)

    for i in range(len(X)):
        match = "✅" if y_pred[i][0] == y[i][0] else "❌"
        print(f"  │   {X[i][0]}   │   {X[i][1]}   │    {y[i][0]}     │  {y_pred_raw[i][0]:.4f}  │   {match}    │")

    print("  └───────┴───────┴──────────┴──────────┴─────────┘")
    print(f"\n  Final Accuracy: {nn.accuracy(X, y) * 100:.1f}%")

    nn.visualize_decision_boundary("XOR Decision Boundary (learned by network)")
    nn.plot_loss_ascii()


# ─────────────────────────────────────────────
#  DEMO 2: Binary Classification
# ─────────────────────────────────────────────

def demo_classification():
    """
    Classifies 2D points into two classes using a neural network.
    Data generated synthetically — no datasets needed.
    """
    print("╔" + "═" * 56 + "╗")
    print("║  DEMO 2: Binary Classification on Synthetic Data        ║")
    print("╚" + "═" * 56 + "╝\n")

    np.random.seed(7)
    n = 200

    # Class 0: points near (0.25, 0.25)
    X0 = np.random.randn(n // 2, 2) * 0.15 + [0.25, 0.25]
    y0 = np.zeros((n // 2, 1))

    # Class 1: points near (0.75, 0.75)
    X1 = np.random.randn(n // 2, 2) * 0.15 + [0.75, 0.75]
    y1 = np.ones((n // 2, 1))

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    # Shuffle
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    # Normalize to [0, 1]
    X = np.clip(X, 0, 1)

    # Train/Test split (80/20)
    split   = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Dataset   : {n} samples (2 classes)")
    print(f"  Train set : {split} samples")
    print(f"  Test set  : {n - split} samples\n")

    nn = NeuralNetwork(layer_sizes=[2, 16, 8, 1], learning_rate=0.3)
    nn.train(X_train, y_train, epochs=3000)

    train_acc = nn.accuracy(X_train, y_train) * 100
    test_acc  = nn.accuracy(X_test,  y_test)  * 100

    print(f"  📊 Train Accuracy : {train_acc:.2f}%")
    print(f"  📊 Test  Accuracy : {test_acc:.2f}%")

    nn.visualize_decision_boundary("Binary Classification — Learned Boundary")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗            ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║            ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║            ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║            ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗       ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝       ║
║                                                              ║
║         NEURAL NETWORK FROM SCRATCH — by Ammar              ║
║         Pure NumPy • No Frameworks • 100% Manual            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    time.sleep(0.5)
    demo_xor()

    time.sleep(0.3)
    demo_classification()

    print("\n" + "═" * 58)
    print("  ✨ Done! This neural network was built from ZERO.")
    print("  📁 GitHub: github.com/ammar")
    print("  🧠 No PyTorch. No TensorFlow. Just math + NumPy.")
    print("═" * 58 + "\n")
