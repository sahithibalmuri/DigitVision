# DigitVision AI: Handwritten Digit Recognizer

DigitVision AI is a highly interactive, premium educational sandbox for exploring image processing and Convolutional Neural Networks (CNNs). Built with **PyTorch**, **Pillow**, and standard **Python HTTP Server** libraries, the application lets you draw handwritten digits in a web browser, get real-time classifications, examine exactly what internal filters "see" at each convolutional layer, and train custom models from scratch directly from a glassmorphic dark-mode dashboard.

---

## Key Features

1. **Interactive Drawing Board**: Canvas drawing utilizing mouse/touch events, support for custom brush widths, and live inference.
2. **Probability Matrix**: Glowing cyber-cyan progress bars showing classification confidence levels (0-9) dynamically.
3. **Live CNN Feature Maps**: Real-time extraction and visualization of activations from Convolutional Layer 1 (edge detection) and Convolutional Layer 2 (loop, angle, and shape detection) for your drawn digit.
4. **Interactive Sandbox Training**: Custom configure hyperparameters (Optimizer, Learning Rate, Network Complexity, Dataset size, and Epoch count) and trigger a multi-epoch training process in the background.
5. **Real-time Metrics Charting**: Interactive dual-axis graph plotting loss reduction and accuracy improvement sweep-by-sweep in Chart.js.

---

## Deep Learning Architecture

The underlying CNN (`MNISTCNN` in PyTorch) consists of:
* **Conv Layer 1**: 16 filters (3x3 kernel), Batch Normalization, ReLU activation, Max-Pooling (2x2) -> Reduces resolution to `14x14`.
* **Conv Layer 2**: 32 filters (3x3 kernel), Batch Normalization, ReLU activation, Max-Pooling (2x2) -> Reduces resolution to `7x7`.
* **Fully Connected 1**: 128 hidden linear units with ReLU activation and Dropout regularization (25%).
* **Fully Connected 2 (Output)**: 10 linear logits, fed into a Softmax activation to compute raw probabilities.

---

## Installation & Setup

No heavy frameworks (like Flask or FastAPI) are required. The server runs fully out-of-the-box.

### Dependencies
Ensure you have Python 3.8+ and PyTorch installed:
```bash
pip install torch torchvision pillow
```

### Run Locally
1. Clone this repository to your system.
2. Run the main backend entrypoint:
   ```bash
   python DigitVision.py
   ```
3. Open your browser and navigate to:
   **[http://localhost:5000/](http://localhost:5000/)**

*If port `5000` is currently occupied by another program, DigitVision will automatically bind to the first available port (5001, 5002, etc.).*

> **Auto-Training**: If no pre-trained weights (`mnist_cnn.pt`) are found in the root directory on startup, the server automatically downloads the MNIST dataset and spins up a background thread to train a default model (takes ~5 seconds), ensuring the drawing canvas works instantly!
