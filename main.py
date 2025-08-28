from neural_network.neural_network import NeuralNetwork
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

def main():

    data = load_digits()
    X = data.data
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    nn = NeuralNetwork.initialize(
        input_size=64,
        hidden_layer_sizes=[100],
        output_size=3,
        eta=0.01,
    )

    nn.train(
        epochs=1000,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )



if __name__ == "__main__":
    main()

