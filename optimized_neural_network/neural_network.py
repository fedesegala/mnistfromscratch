from dataclasses import dataclass
import numpy as np
from typing import List, Optional
from multiprocessing import Pool

@dataclass
class NeuralNetwork:
    input_size: int
    hidden_layer_sizes: List[int]
    output_size: int
    w:          List[np.ndarray]
    V:          List[List[np.ndarray]]
    h:          List[List[np.ndarray]]
    d:          List[np.ndarray]
    lr:         float
    x_train:    Optional[np.ndarray] = None
    x_test:     Optional[np.ndarray] = None
    y_train:    Optional[np.ndarray] = None
    y_test:     Optional[np.ndarray] = None


    @classmethod
    def initialize(cls, input_size: int, hidden_layer_sizes: List[int], output_size:int, lr: float) -> "NeuralNetwork":
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
            lr=lr,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h = []
        self.V = []

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
        computed_delta = compute_single_delta_task(self.h, self.V, self.w, y)

        for d in reversed(computed_delta):
            self.d.append(d)

        return self.d

    def update_weights(self):
        for m in reversed(range(len(self.V))):
            if m == 0:
                break
            v = np.append(self.V[m-1], -1)
            self.w[m-1] += self.lr * np.outer(self.d[m-1], v)

    def train(self, epochs, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test = x_train, x_test

        n_classes = len(np.unique(y_train))
        self.y_train = one_hot_encode(y_train, n_classes)
        self.y_test = one_hot_encode(y_test, n_classes)

        for epoch in range(epochs):
            for x, y in zip (self.x_train, self.y_train):
                self.forward(x)
                self.compute_deltas(y)
                self.update_weights()

            if epoch % 10 == 0:
                corrects = 0

                for test_sample, test_target in zip(x_test, y_test):
                    self.V = []
                    self.h = []
                    self.d = []

                    forward_result = self.forward(test_sample)
                    y_pred = np.argmax(forward_result)

                    if y_pred == test_target:
                        corrects += 1

                print(f"Epoch: {epoch}: current accuracy: {corrects / len(y_test)}")


def compute_single_delta_task(local_h, local_v, weights, target):
    d = []

    for m in reversed(range(len(local_v))):
        if m == 0:
            break
        if m  == len(local_v) - 1:
            d.append(
                (target - local_v[m]) * _sigmoid_derivative(local_h[m-1])
            )
        else:
            delta_hat = _sigmoid_derivative(local_h[m-1])
            w_vector = weights[m][:,:-1].T
            previous_delta = d[-1]

            dot_prod = np.dot(w_vector, previous_delta)
            d.append(
                delta_hat * dot_prod
            )

    return d

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))
def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    sigmoid_output = _sigmoid(z)
    return sigmoid_output * (1 - sigmoid_output)

def one_hot_encode(target: np.ndarray, n_classes: int):
    encoding = np.eye(n_classes)[target]
    return encoding