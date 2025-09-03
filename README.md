# Neural Network From Scratch

This project focuses on implementing a **neural network from scratch** in Python, optionally including **pruning** techniques to reduce model complexity.  
The network has been tested on the following datasets:

- `load_iris` (Iris flower classification)  
- `load_digits` (Handwritten digits 8x8)  
- `MNIST` (Handwritten digits 28x28)  

The goal is to understand the mechanics of neural networks without relying on high-level libraries like TensorFlow or PyTorch.

---

## 🔹 1. Running the Project with **uv** (recommended)

### Install uv
If `uv` is not installed, you can install it with `pip`:

```bash
pip install uv
```

Or using the official installer: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
``bash
uv sync
``

### Run the project
```bash
uv run python main.py
```
## 🔹 2. Running the Project with a traditional virtual environment
### Create a virtual environment
```bash
python -m venv .venv
```
### Activate it: 
```bash
python -m venv .venv
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the project
```bash
python main.py
```

