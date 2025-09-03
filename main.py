
from optimized_neural_network.neural_network import NeuralNetwork as OptimizedNeuralNetwork
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def compare_on_toy_dataset(toy_name = "digits"):
    if toy_name == "digits":
        data = load_digits()
    else:
        data = load_iris()
    X = data.data
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    print(f"x_train shape: {x_train.shape}")
    # DIGITS
    small_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[20,20] if toy_name == "digits" else [2],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    medium_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50, 50] if toy_name == "digits" else [5],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    large_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[100, 100] if toy_name == "digits" else [20],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    epochs_number = 30 if toy_name == "digits" else 100
    small_net.train(
        epochs=epochs_number,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    medium_net.train(
        epochs=epochs_number,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    large_net.train(
        epochs=epochs_number,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    stats_small = small_net.stats
    stats_medium = medium_net.stats
    stats_large = large_net.stats

    plot_comparison(
        [
            stats_small,
            stats_medium,
            stats_large,
        ],
        [
            f"Small Model: hidden units {small_net.hidden_layer_sizes}",
            f"Medium Model: hidden units {medium_net.hidden_layer_sizes}",
            f"Large Model: hidden units {large_net.hidden_layer_sizes}",
        ]
    )

def compare_with_mnist():
    print("Downloading MNIST dataset")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, Y = np.array(mnist["data"], dtype=np.float64), mnist["target"].astype(int)
    X /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

    print(f"x_train shape: {x_train.shape}")

    small_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[10],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    medium_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    large_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[100],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    multilayer_net = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[50,20,15],
        output_size=len(np.unique(y_test)),
        lr=0.03,
    )

    small_net.train(
        epochs=15,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    medium_net.train(
        epochs=15,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    large_net.train(
        epochs=15,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    multilayer_net.train(
        epochs=15,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )



    stats_small = small_net.stats
    stats_medium = medium_net.stats
    stats_large = large_net.stats
    stats_multi = multilayer_net.stats

    plot_comparison(
        [
            stats_small,
            stats_medium,
            stats_large,
            stats_multi,
        ],
        [
            f"Small Model: hidden units {small_net.hidden_layer_sizes}",
            f"Medium Model: hidden units {medium_net.hidden_layer_sizes}",
            f"Large Model: hidden units {large_net.hidden_layer_sizes}",
            f"Multilayer Model: hidden units {multilayer_net.hidden_layer_sizes}"
        ]
    )

def pruning_load_digits():
    data = load_digits()
    X = data.data
    Y = data.target

    n_classes = len(np.unique(Y))

    initial_model_size = 100

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    print(f"x_train shape: {x_train.shape}")
    # DIGITS
    nn = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[initial_model_size],
        output_size=len(np.unique(y_test)),
        lr=0.01,
    )

    nn.train(
        epochs=5,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    stats_initial = nn.stats
    initial_hidden_units = nn.hidden_layer_sizes[0]

    history = nn.prune_iterative(epsilon=0.01, max_prunes=80)

    print(nn.hidden_layer_sizes)

    print(nn.compute_stats_on_sample(x_train, nn.one_hot_encode(y_train)))
    print(nn.compute_stats_on_sample(x_test, nn.one_hot_encode(y_test)))

    target_model_size = initial_model_size - len(history)
    small_model = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[target_model_size],
        output_size=n_classes,
        lr=0.01,
    )

    small_model.train(
        epochs=5,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    small_stats = small_model.stats

    train_loss, train_acc = nn.compute_stats_on_sample(x_train, nn.one_hot_encode(y_train))
    test_loss, test_acc = nn.compute_stats_on_sample(x_test, nn.one_hot_encode(y_test))

    plot_comparison(
        [
            small_stats,
            stats_initial,
        ],
        [
            f"Small model: hidden units {small_model.hidden_layer_sizes}",
            f"Large model before pruning: hidden units [{initial_hidden_units}]",
        ],
        pruning_train_acc=train_acc,
        pruning_test_acc=test_acc,
        pruning_train_loss=train_loss,
        pruning_test_loss=test_loss,
    )

def pruning_mnist():
    print("Downloading MNIST dataset")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, Y = np.array(mnist["data"], dtype=np.float64), mnist["target"].astype(int)
    X /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

    print(f"x_train shape: {x_train.shape}")

    initial_model_size = 100

    # DIGITS
    nn = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[initial_model_size],
        output_size=len(np.unique(y_test)),
        lr=0.01,
    )

    nn.train(
        epochs=5,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    stats_initial = nn.stats
    initial_hidden_units = nn.hidden_layer_sizes[0]

    history = nn.prune_iterative(epsilon=0.01, max_prunes=80)

    print(nn.hidden_layer_sizes)

    print(nn.compute_stats_on_sample(x_train, nn.one_hot_encode(y_train)))
    print(nn.compute_stats_on_sample(x_test, nn.one_hot_encode(y_test)))

    target_model_size = initial_model_size - len(history)
    small_model = OptimizedNeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[target_model_size],
        output_size=len(np.unique(y_test)),
        lr=0.01,
    )

    small_model.train(
        epochs=5,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        compute_stats_interval=1
    )

    stats_small = small_model.stats

    train_loss, train_acc = nn.compute_stats_on_sample(x_train, nn.one_hot_encode(y_train))
    test_loss, test_acc = nn.compute_stats_on_sample(x_test, nn.one_hot_encode(y_test))

    plot_comparison(
        [
            stats_small,
            stats_initial,
        ],
        [
            f"Small model: hidden units {small_model.hidden_layer_sizes}",
            f"Large model before pruning: hidden units [{initial_hidden_units}]",
        ],
        pruning_train_acc=train_acc,
        pruning_test_acc=test_acc,
        pruning_train_loss=train_loss,
        pruning_test_loss=test_loss,
    )

def plot_comparison(stats, titles,
                    pruning_train_acc=0,
                    pruning_test_acc=0,
                    pruning_train_loss=0,
                    pruning_test_loss=0):
    # Create an empty list to store all data
    data = []

    # Prepare the data for plotting
    for i, (stat, title) in enumerate(zip(stats, titles)):
        for epoch_data in stat:
            epoch = epoch_data['epoch']
            data.append({
                'Epoch': epoch,
                'Training Accuracy': epoch_data['train_acc'],
                'Training Loss': epoch_data['train_loss'],
                'Test Accuracy': epoch_data['test_acc'],
                'Test Loss': epoch_data['test_loss'],
                'Model': title
            })

    # Convert the list of dictionaries into a DataFrame for Seaborn
    df = pd.DataFrame(data)

    # Trova il range delle epoche (min e max)
    min_epoch, max_epoch = df['Epoch'].min(), df['Epoch'].max()

    # Set the figure size
    plt.figure(figsize=(18, 10))

    # Plot Training Accuracy
    plt.subplot(221)
    sns.lineplot(x='Epoch', y='Training Accuracy', hue='Model', data=df)
    if pruning_train_acc != 0:
        plt.hlines(y=pruning_train_acc, xmin=min_epoch, xmax=max_epoch,
                   color='r', linestyle='--',
                   label=f'Pruning Train Acc = {pruning_train_acc:.2f}')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)

    # Plot Training Loss
    plt.subplot(222)
    sns.lineplot(x='Epoch', y='Training Loss', hue='Model', data=df)
    if pruning_train_loss != 0:
        plt.hlines(y=pruning_train_loss, xmin=min_epoch, xmax=max_epoch,
                   color='g', linestyle='--',
                   label=f'Pruning Train Loss = {pruning_train_loss:.4f}')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Test Accuracy
    plt.subplot(223)
    sns.lineplot(x='Epoch', y='Test Accuracy', hue='Model', data=df)
    if pruning_test_acc != 0:
        plt.hlines(y=pruning_test_acc, xmin=min_epoch, xmax=max_epoch,
                   color='r', linestyle='--',
                   label=f'Pruning Test Acc = {pruning_test_acc:.2f}')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)

    # Plot Test Loss
    plt.subplot(224)
    sns.lineplot(x='Epoch', y='Test Loss', hue='Model', data=df)
    if pruning_test_loss != 0:
        plt.hlines(y=pruning_test_loss, xmin=min_epoch, xmax=max_epoch,
                   color='g', linestyle='--',
                   label=f'Pruning Test Loss = {pruning_test_loss:.4f}')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    compare_on_toy_dataset("iris")
    compare_on_toy_dataset("digits")
    compare_with_mnist()
    pruning_load_digits()
    pruning_mnist()