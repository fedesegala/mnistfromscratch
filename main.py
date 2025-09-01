import drawdata

from optimized_neural_network.neural_network import NeuralNetwork as OptimizedNeuralNetwork
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

def compare_with_toy_datasets(toy_name = "digits"):
    # print("Downloading MNIST dataset")
    # mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    # X, Y = np.array(mnist["data"], dtype=np.float64), mnist["target"].astype(int)
    # X /= 255.0

    if toy_name == "digits":
        data = load_digits()
    else:
        data = load_iris()
    X = data.data
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    print(f"x_train shape: {x_train.shape}")
    # DIGITS
    nn_momentum = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50,50] if toy_name == "digits" else [5],
        output_size=len(np.unique(y_test)),
        lr=0.01,
        momentum=0.75 if toy_name == "digits" else 0.8,
    )

    nn_momentum.train(
        epochs=30,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    nn_non_momentum = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50,50] if toy_name == "digits" else [5],
        output_size=len(np.unique(y_test)),
        lr=0.01,
        momentum=0
    )

    nn_non_momentum.train(
        epochs=30,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    momentum_stats = nn_momentum.stats
    non_momentum_stats = nn_non_momentum.stats

    plot_comparison(
        momentum_stats,
        non_momentum_stats,
        f"Model: lr{nn_momentum.lr}, alpha: {nn_momentum.momentum}",
        f"Model: lr{nn_momentum.lr}, alpha: {nn_non_momentum.momentum}",

    )

def compare_with_mnist():
    print("Downloading MNIST dataset")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, Y = np.array(mnist["data"], dtype=np.float64), mnist["target"].astype(int)
    X /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

    print(f"x_train shape: {x_train.shape}")

    nn_momentum = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50, 50],
        output_size=len(np.unique(y_test)),
        lr=0.4,
        momentum=0.9
    )

    nn_momentum.train(
        epochs=30,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    nn_non_momentum = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50, 50],
        output_size=len(np.unique(y_test)),
        lr=0.4,
        momentum=0
    )

    nn_non_momentum.train(
        epochs=30,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    momentum_stats = nn_momentum.stats
    non_momentum_stats = nn_non_momentum.stats

    plot_comparison(
        momentum_stats,
        non_momentum_stats,
        f"Model: lr{nn_momentum.lr}, alpha: {nn_momentum.momentum}",
        f"Model: lr{nn_momentum.lr}, alpha: {nn_non_momentum.momentum}",
    )


def plot_comparison(stats1, stats2, title1, title2):
    epochs = [s['epoch'] for s in stats1]

    plt.figure(figsize=(18, 6))  # Increased figure size for 4 subplots

    # 1. Training Accuracy
    plt.subplot(221)
    plt.plot(epochs, [s['train_acc'] for s in stats1], label=f'{title1} - Train Acc.')
    plt.plot(epochs, [s['train_acc'] for s in stats2], label=f'{title2} - Train Acc.')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 2. Training Loss
    plt.subplot(222)
    plt.plot(epochs, [s['train_loss'] for s in stats1], label=f'{title1} - Train Loss')
    plt.plot(epochs, [s['train_loss'] for s in stats2], label=f'{title2} - Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 3. Test Accuracy
    plt.subplot(223)
    plt.plot(epochs, [s['test_acc'] for s in stats1], label=f'{title1} - Test Acc.')
    plt.plot(epochs, [s['test_acc'] for s in stats2], label=f'{title2} - Test Acc.')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 4. Test Loss
    plt.subplot(224)
    plt.subplot(224)
    plt.plot(epochs, [s['test_loss'] for s in stats1], label=f'{title1} - Test Loss')
    plt.plot(epochs, [s['test_loss'] for s in stats2], label=f'{title2} - Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    compare_with_toy_datasets("digits")
    # compare_with_mnist()