from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class NeuralNetwork:
    input_size: int
    hidden_layer_size: int # todo: try to add more hidden layers
    output_size: int
    weights: List[np.ndarray]

    @classmethod
    def initialize(cls, input_size: int, hidden_layer_size: int, output_size:int) -> "NeuralNetwork":
        w_jk = np.random.randn(hidden_layer_size, input_size + 1) / np.sqrt(hidden_layer_size)
        w_ij = np.random.randn(output_size, hidden_layer_size + 1) / np.sqrt(output_size)

        return cls(
            input_size=input_size,
            hidden_layer_size=hidden_layer_size,
            output_size=output_size,
            weights=[w_jk, w_ij]
        )
