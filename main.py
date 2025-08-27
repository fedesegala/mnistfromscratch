from neural_network.neural_network import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork.initialize(
        input_size=4,
        hidden_layer_size=3,
        output_size=2
    )