from optimized_neural_network.neural_network import NeuralNetwork
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # print("Downloading MNIST dataset")
    # mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    # X, Y = np.array(mnist["data"], dtype=np.float64), mnist["target"].astype(int)
    # X /= 255.0

    data = load_digits()
    X = data.data
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    print(f"x_train shape: {x_train.shape}")
    # DIGITS
    nn = NeuralNetwork.initialize(
        input_size=x_train.shape[-1],
        hidden_layer_sizes=[100],
        output_size=len(np.unique(y_test)),
        lr=0.01,
    )

    nn.train(
        epochs=200,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()

