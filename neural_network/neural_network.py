from dataclasses import dataclass
import numpy as np
from typing import List
from collections import defaultdict

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))
def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    sigmoid_output = _sigmoid(z)
    return sigmoid_output * (1 - sigmoid_output)

def one_hot_encode(target, n_classes):
    zeros = np.zeros(n_classes)
    zeros[target] = 1

    return zeros


@dataclass
class NeuralNetwork:
    input_size: int
    hidden_layer_sizes: List[int]
    output_size: int
    w: List[np.ndarray]
    V: List[np.ndarray]
    h: List[np.ndarray]
    d: List[np.ndarray]
    lr: float

    @classmethod
    def initialize(cls, input_size: int, hidden_layer_sizes: List[int], output_size:int, eta: float) -> "NeuralNetwork":
        weights = []
        previous_size = input_size
        for h_size in hidden_layer_sizes:
            weights.append(np.random.randn(h_size, previous_size + 1) / np.sqrt(h_size))
            previous_size = h_size
        weights.append(np.random.randn(output_size, previous_size + 1) / np.sqrt(output_size))

        return cls(
            input_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=output_size,
            w=weights,
            V=[],
            h=[],
            d=[],
            lr=eta,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.V = []
        self.h = []

        self.V.append(x)

        for i,layer in enumerate(self.w):
            x = np.append(x, -1.0)
            h_i = layer.dot(x) # z = w*x
            self.h.append(h_i)
            x = _sigmoid(h_i)
            self.V.append(x)

        return self.V[-1]

    def compute_deltas(self, y: np.ndarray) -> List[np.ndarray]:
        self.d = []
        for m in reversed(range(len(self.V))):
            if m == 0:
                break
            local_d = np.array([])
            if m == len(self.V) - 1:    # update hidden to output weights
                for i,V_m_i in enumerate(self.V[m]):
                    local_d = np.append(local_d, _sigmoid_derivative(self.h[m-1][i]) * (y[i] - V_m_i))
            else:
                for i, V_m_i in enumerate(self.V[m]):
                    delta_hat_i = _sigmoid_derivative(self.h[m-1][i])
                    s = 0
                    previous_delta = self.d[-1]
                    previous_layer_weights = self.w[m][:,i]

                    for j, d in enumerate(previous_delta):
                        if len(previous_layer_weights) == j:
                            print("")
                        s += d * previous_layer_weights[j]

                    local_d = np.append(local_d, delta_hat_i * s)
            self.d.append(local_d)


        self.d = [d for d in reversed(self.d)]
        return self.d

    def update_weights(self):
        for m in reversed(range(len(self.V))):
            if m == 0:
                break
            for i in range(len(self.w[m-1])):
                for j in range(len(self.w[m-1][i])):
                    if j == len(self.w[m-1][i]) - 1:
                        update = self.lr * self.d[m-1][i] * -1
                    else:
                        update = self.lr * self.d[m-1][i] * self.V[m-1][j]
                    self.w[m-1][i][j] += update

    def train(self, epochs, x_train, x_test, y_train, y_test):
        n_classes = len(np.unique(y_train))

        for epoch in range(epochs):
            for input_sample, target in zip(x_train, y_train):
                self.forward(input_sample)
                self.compute_deltas(one_hot_encode(target, n_classes))
                self.update_weights()

            if epoch % 10 == 0:
                corrects = 0
                for test_sample, test_target in zip(x_test, y_test):
                    y_pred = np.argmax(self.forward(test_sample))
                    if y_pred == test_target:
                        corrects += 1

                print(f"Epoch: {epoch}: current accuracy: {corrects / len(y_test)}")


# for i in reversed(range(len(self.V))):
#     if i == 0:
#         break
#     if i == len(self.V) - 1:
#         error = y - self.V[i]
#         # print(f"error at layer {i}: {error}")
#         delta = error * _sigmoid_derivative(self.V[i])
#         # print(f"delta at layer {i}: {delta}")
#         self.d.append(delta)
#     else:
#         next_weights = self.w[i]  # weights from this layer to next
#         next_delta = self.d[-1]  # delta from the next layer (we are appending in reverse order)
#         error = next_weights.T @ next_delta
#         # print(f"error at layer {i}: {error}")
#         delta = error[:-1] * _sigmoid_derivative(self.h[i - 1])
#         # print(f"delta at layer {i}: {delta}")
#         self.d.append(delta)
