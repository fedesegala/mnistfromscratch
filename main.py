from neural_network.neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":
    nn = NeuralNetwork.initialize(
        input_size=4,
        hidden_layer_size=3,
        output_size=2
    )

    response = (nn.forward([1,1,1,1]))

    for layer_response in response:
        print(layer_response)
