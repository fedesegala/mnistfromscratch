from optimized_neural_network.neural_network import NeuralNetwork
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

def main():

    data = load_iris()
    X = data.data
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    nn = NeuralNetwork.initialize(
        input_size=64,
        hidden_layer_sizes=[50,20,15],
        output_size=10,
        lr=0.01,
        batch_size=20
    )

    nn = NeuralNetwork.initialize(
        input_size=4,
        hidden_layer_sizes=[10],
        output_size=3,
        lr=0.01,
        batch_size=150,
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

