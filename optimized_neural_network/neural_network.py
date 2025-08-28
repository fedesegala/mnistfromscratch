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
    batch_size: int
    x_train:    Optional[np.ndarray] = None
    x_test:     Optional[np.ndarray] = None
    y_train:    Optional[np.ndarray] = None
    y_test:     Optional[np.ndarray] = None
    current_batch_expected: List[np.ndarray] = None


    @classmethod
    def initialize(cls, input_size: int, hidden_layer_sizes: List[int], output_size:int, lr: float, batch_size: int = 3) -> "NeuralNetwork":
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
            batch_size=batch_size,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h.append([])
        self.V.append([])
        self.V[-1].append(x)

        for i,layer in enumerate(self.w):
            x = np.append(x, -1.0)
            h_i = layer.dot(x) # z = w*x
            self.h[-1].append(h_i)
            x = _sigmoid(h_i)
            self.V[-1].append(x)

        return self.V

    def compute_deltas(self, y: np.ndarray) -> List[np.ndarray]:
        jobs = []
        self.d = []

        for input_sample_idx, sample_target in enumerate(self.current_batch_expected):
            parallel_h = self.h[input_sample_idx]
            parallel_v = self.V[input_sample_idx]

            jobs.append((parallel_h, parallel_v, self.w, sample_target))

        with Pool() as pool:
            computed_deltas = pool.starmap(compute_single_delta_task, jobs)

        for i, computed_delta in enumerate(computed_deltas):
            self.d.append([d for d in reversed(computed_delta)])

        return self.d

    def update_weights(self):
        for i in range(len(self.V)):
            V = self.V[i]
            d = self.d[i]
            for m in reversed(range(len(V))):
                if m == 0:
                    break
                v = np.append(V[m-1], -1)
                self.w[m-1] += self.lr * np.outer(d[m-1], v)

    def train(self, epochs, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test = x_train, x_test

        n_classes = len(np.unique(y_train))
        self.y_train = one_hot_encode(y_train, n_classes)
        self.y_test = one_hot_encode(y_test, n_classes)

        for epoch in range(epochs):
            for start in range(0, len(x_train), self.batch_size):
                end = start + self.batch_size
                batch_x = self.x_train[start:end]
                batch_y = self.y_train[start:end]

                self.current_batch_expected = []
                self.V = []
                self.h = []

                for x, y in zip(batch_x, batch_y):
                    self.current_batch_expected.append(y)
                    self.forward(x)
                self.compute_deltas(batch_y)
                self.update_weights()

            if epoch % 10 == 0:
                corrects = 0

                for test_sample, test_target in zip(x_test, y_test):
                    self.V = []
                    self.h = []
                    self.d = []

                    forward_result = self.forward(test_sample)[0][-1]
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