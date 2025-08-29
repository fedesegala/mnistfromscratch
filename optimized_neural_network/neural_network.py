from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics import mean_squared_error, accuracy_score

@dataclass
class NeuralNetwork:
    input_size: int
    hidden_layer_sizes: List[int]
    output_size: int
    w:          List[np.ndarray]
    V:          List[List[np.ndarray]]
    h:          List[List[np.ndarray]]
    d:          List[np.ndarray]
    previous_update: List[np.ndarray]
    lr:         float
    momentum:   float
    stats:      List[Dict] = None
    x_train:    Optional[np.ndarray] = None
    x_test:     Optional[np.ndarray] = None
    y_train:    Optional[np.ndarray] = None
    y_test:     Optional[np.ndarray] = None
    test_loss: List[float] = None



    @classmethod
    def initialize(cls, input_size: int, hidden_layer_sizes: List[int], output_size:int, lr: float, momentum: float = 0) -> "NeuralNetwork":
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
            previous_update=[],
            lr=lr,
            momentum=momentum,
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

    # this method shall not be used for training only for getting faster predictions
    def batch_forward(self, batch_x):
        for layer in self.w:
            batch_x = np.vstack((batch_x, np.ones((1,batch_x.shape[-1]))))
            h_i = layer.dot(batch_x)
            batch_x = _sigmoid(h_i)

        return batch_x

    def compute_deltas(self, y: np.ndarray) -> List[np.ndarray]:
        self.d = []
        computed_delta = compute_delta(self.h, self.V, self.w, y)

        for d in reversed(computed_delta):
            self.d.append(d)

        return self.d

    def update_weights(self):
        previous_iteration_update = self.previous_update.copy()
        current_iteration_update = []
        for m in reversed(range(len(self.V))):
            if m == 0:
                break
            v = np.append(self.V[m-1], -1)
            if len(previous_iteration_update) == 0:
                current_level_update = self.lr * np.outer(self.d[m-1],v)
                current_iteration_update.append(current_level_update)
                self.w[m-1] += current_level_update
            else:
                current_level_update = self.lr * np.outer(self.d[m-1],v) + self.momentum * previous_iteration_update[m-1]
                current_iteration_update.append(current_level_update)
                self.w[m - 1] += current_level_update

        self.previous_update = [u for u in reversed(current_iteration_update)]

    def train(self, epochs, x_train, x_test, y_train, y_test, compute_stats_interval: int = 50):
        self.x_train, self.x_test = x_train, x_test

        n_classes = len(np.unique(y_train))
        self.y_train = one_hot_encode(y_train, n_classes)
        self.y_test = one_hot_encode(y_test, n_classes)

        self.test_loss = []

        for epoch in range(epochs):
            if self.stats is None:
                self.stats = []

            permutation = np.random.permutation(len(x_train))
            self.x_train, self.y_train = self.x_train[permutation], self.y_train[permutation]

            for i, (x, y) in enumerate(zip(self.x_train, self.y_train)):
                self.forward(x)
                self.compute_deltas(y)
                self.update_weights()

                if epoch == 0 and i % 200 == 0:
                    corrects = 0
                    for test_sample, test_target in zip(x_test[:1000], y_test[:1000]):
                        self.V = []
                        self.h = []
                        self.d = []

                        forward_result = self.forward(test_sample)
                        y_pred = np.argmax(forward_result)

                        if y_pred == test_target:
                            corrects += 1

                    self.test_loss.append(corrects/len(y_test[:1000]))

            if epoch % compute_stats_interval == 0:
                y_pred_train = self.batch_forward(self.x_train.T)
                y_pred_test = self.batch_forward(self.x_test.T)

                train_loss = mean_squared_error(self.y_train, y_pred_train.T)
                test_loss = mean_squared_error(self.y_test, y_pred_test.T)

                y_pred_train = np.eye(10)[np.argmax(y_pred_train, axis=0)]
                y_pred_test = np.eye(10)[np.argmax(y_pred_test, axis=0)]

                train_acc = accuracy_score(self.y_train, y_pred_train)
                test_acc = accuracy_score(self.y_test, y_pred_test)

                self.stats.append(
                    {
                        "epoch": epoch,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "test_acc": test_acc,
                        "test_loss": test_loss
                    }
                )

                print(f"Epoch: {epoch}:\n\ttrain loss: {train_loss:.4f}\t train accuracy: {train_acc*100:.2f}%\n\ttest loss: {test_loss:.4f}\t\ttest accuracy: {test_acc*100:.2f}%")

    def predict(self, sample):
        self.V = []
        self.h = []
        self.d = []

        return np.argmax(self.forward(sample))

    def predict_probs(self, sample):
        self.V = []
        self.h = []
        self.d = []

        return self.forward(sample)

def compute_delta(local_h, local_v, weights, target):
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