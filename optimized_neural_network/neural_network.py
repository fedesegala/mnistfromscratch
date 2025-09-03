from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics import mean_squared_error, accuracy_score
from activations.sigmoid import sigmoid, sigmoid_derivative

@dataclass
class NeuralNetwork:
    input_size: int
    hidden_layer_sizes: List[int]
    output_size: int
    w: List[np.ndarray]
    V: List[List[np.ndarray]]
    h: List[List[np.ndarray]]
    d: List[np.ndarray]
    lr: float
    stats: List[Dict] = None
    x_train: Optional[np.ndarray] = None
    x_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    test_loss: List[float] = None
    n_classes: int = 0

    @classmethod
    def initialize(cls, input_size: int, hidden_layer_sizes: List[int], output_size: int, lr: float) -> "NeuralNetwork":
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
        """
        Performs the forward pass on a single sample in the network populating the V and h attributes
        for each layer the input crosses

        :param x: the input sample (shape: (n_features,))
        :return: the output of the last layer of the network
        """
        self.h = []
        self.V = []

        self.V.append(x)

        for i, layer in enumerate(self.w):
            x = np.append(x, -1.0)      # add bias
            h_i = layer.dot(x)          # h_i = sum_j(w_ij * V_j)
            self.h.append(h_i)
            x = sigmoid(h_i)            # V = g(h)
            self.V.append(x)

        return self.V[-1]

    def batch_forward(self, batch_x):
        """
        Performs the forward pass on a batch of samples in the network
        Important note: this method shall only be used for inference, not training, since
        it does not populate the V and h attributes

        :param batch_x: the input batch (shape: (n_features, n_samples))
        :return: the output of the last layer of the network (shape: (n_classes, n_samples))
        """
        for layer in self.w:
            batch_x = np.vstack((batch_x, -np.ones((1, batch_x.shape[-1]))))
            h_i = layer.dot(batch_x)
            batch_x = sigmoid(h_i)

        return batch_x

    def compute_deltas(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Invokes a utility function to compute the delta vectors for each layer and populates the d attribute
        given the expected output y (one hot encoded)

        :param y: the one-hot encoded expected output (shape: (n_classes,))
        :return:
        """
        self.d = []
        computed_delta = self.compute_delta(y)

        for d in reversed(computed_delta):
            self.d.append(d)

    def compute_delta(self, target) -> List[np.ndarray]:
        """
        Given a target vector (one hot encoded) computes the delta vectors to update the weights in each layer

        :param target: the expected output (one hot encoded)
        :return: a list of delta vectors, one for each layer
        """
        d = []

        # for each layer of the network in reversed order
        for m in reversed(range(len(self.V))):
            if m == 0:  # input layer, no delta to compute
                break
            if m == len(self.V) - 1:    # hidden to output update: (y - y_pred) * g'(h)
                d.append(
                    (target - self.V[m]) * sigmoid_derivative(self.h[m - 1])    # h has one element less than V, hence m-1
                )
            else:   # hidden to hidden update w_jk
                delta_hat = sigmoid_derivative(self.h[m - 1])   # g'(h_j)
                w_ij = self.w[m][:, :-1].T  # weight between current layer and next
                delta_i = d[-1]          # error made by next layer

                dot_prod = np.dot(w_ij, delta_i) # weighted average of next layer's deltas
                d.append(
                    delta_hat * dot_prod
                )

        return d

    def update_weights(self):
        """
        Performs the actual weights update after computations of delta vectors
        using learning rate and (heavy ball) momentum.

        :return:
        """
        for m in reversed(range(len(self.V))):
            if m == 0:
                break
            v = np.append(self.V[m - 1], -1)
            self.w[m - 1] += self.lr * np.outer(self.d[m - 1], v)

    def train(self, epochs, x_train, x_test, y_train, y_test, compute_stats_interval: int = 2):
        """
        Performs the training of the network on the given training set for a number of epochs printing training statistics

        :param epochs: number of training epochs
        :param x_train: training set
        :param x_test: test set
        :param y_train: training labels
        :param y_test: test labels
        :param compute_stats_interval: epochs interval to compute and print statistics
        :return:
        """
        self.x_train, self.x_test = x_train, x_test

        self.n_classes = len(np.unique(y_train))
        self.y_train = self.one_hot_encode(y_train)
        self.y_test = self.one_hot_encode(y_test)

        self.test_loss = []

        for epoch in range(epochs):
            if self.stats is None:
                self.stats = []

            # shuffle training set to enuser order of sample does not bias the training
            permutation = np.random.permutation(len(x_train))
            self.x_train, self.y_train = self.x_train[permutation], self.y_train[permutation]

            for i, (x, y) in enumerate(zip(self.x_train, self.y_train)):
                self.forward(x)
                self.compute_deltas(y)
                self.update_weights()

            if epoch % compute_stats_interval == 0:
                train_loss, train_acc = self.compute_stats_on_sample(self.x_train, self.y_train)
                test_loss, test_acc = self.compute_stats_on_sample(self.x_test, self.y_test)

                self.stats.append(
                    {
                        "epoch": epoch,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "test_acc": test_acc,
                        "test_loss": test_loss
                    }
                )

                print(
                    f"Epoch: {epoch}:\n\ttrain loss: {train_loss:.4f}\t train accuracy: {train_acc * 100:.2f}%\n\ttest loss: {test_loss:.4f}\t\ttest accuracy: {test_acc * 100:.2f}%")

    def predict(self, sample):
        """
        Performs a real prediction on a single sample returning the class with highest probability

        :param sample: a single input sample (shape: (n_features,))
        :return:
        """
        self.V = []
        self.h = []
        self.d = []

        return np.argmax(self.forward(sample))

    def compute_stats_on_sample(self, x, y):
        """
        Computes loss and accuracy on a given sample

        :param x: the input sample
        :param y: the expected output (one hot encoded)
        :return: loss and accuracy
        """
        y_pred = self.batch_forward(x.T)

        loss = mean_squared_error(y, y_pred.T)

        y_pred = np.argmax(y_pred, axis=0)
        acc = accuracy_score(np.argmax(y, axis=1), y_pred)

        return loss, acc

    def predict_probs(self, sample):
        """
        Given a sample, returns the output probabilities of each class
        :param sample:
        :return:
        """
        self.V = []
        self.h = []
        self.d = []

        return self.forward(sample)

    def prune_iterative(self, epsilon: float, max_prunes: Optional[int] = None) -> List[Dict]:
        """
        Iteratively prunes hidden units from the network until performance degrades more than epsilon
        or until max_prunes units have been removed (if specified).

        :param epsilon: maximum allowed performance degradation (in terms of MSE increase) on the training set
        :param max_prunes: maximum number of hidden units to prune
        :return: the history of pruning operations performed
        """
        if len(self.hidden_layer_sizes) != 1:
            raise NotImplementedError("Pruning is currently supported only on networks with one hidden layer.")

        history = []
        perf_ref = self.compute_mse(self.x_train, self.y_train)

        while True:
            # save backup of weights so that we can restore to the previous state
            # when pruning will have degraded
            w_backup = [np.copy(w) for w in self.w]
            hsize_backup = self.hidden_layer_sizes.copy()

            removed_hidden_unit = self.prune_single_unit()

            perf_new = self.compute_mse(self.x_train, self.y_train)
            delta_perf = perf_new - perf_ref

            if delta_perf < epsilon:
                # performance did not degrade too much, keep pruning
                perf_ref = perf_new
                history.append({
                    "removed": int(removed_hidden_unit),
                    "perf": float(perf_ref),
                    "nh": int(self.hidden_layer_sizes[0])
                })
                if max_prunes is not None and len(history) >= max_prunes:
                    break
            else:
                # performance degraded too much, restore previous state and stop pruning
                self.w = [np.copy(w) for w in w_backup]
                self.hidden_layer_sizes = hsize_backup
                break

        return history

    def compute_batch_output_of_hidden_layer(self) -> np.ndarray:
        """
        Computes the output of the hidden layer for all training samples

        :return: a matrix of shape (n_samples, n_h) where each row is the output of the hidden layer
        """
        n_samples = self.x_train.shape[0]
        x_with_bias = np.hstack([self.x_train, -np.ones((n_samples, 1))])
        input_to_hidden_weights = self.w[0]

        return sigmoid(x_with_bias @ input_to_hidden_weights.T)

    def compute_mse(self, sample: np.ndarray, vectorial_y: np.ndarray) -> float:
        """
        Computes the Mean Squared Error on the given sample

        :param sample: shape (n_samples, n_features)
        :param vectorial_y: expected output in one-hot encoding, shape (n_samples, n_classes)
        :return: mean error on the sample
        """
        y_pred = self.batch_forward(sample.T).T
        return mean_squared_error(vectorial_y, y_pred)

    def prune_single_unit(self) -> int:
        """
        Prunes a single hidden unit from the network using residual reduction and least squares.
        The hidden unit with lowest contribution is selected for removal. Then updates are computed on
        the weights connecting the remaining hidden units to the output layer to compensate for the removal.

        :return: the index of the removed hidden unit
        """

        hidden_layer_output = self.compute_batch_output_of_hidden_layer()
        hidden_units_count = hidden_layer_output.shape[-1]

        hidden_to_output_weights = self.w[-1]
        output_units_count = hidden_to_output_weights.shape[0]

        ''' --- choose minimum norm hidden unit --- '''
        b_norms = np.zeros(hidden_units_count)
        # range over all hidden units
        for h in range(hidden_units_count):
            output_units_b = []
            # compute b for each output unit (over all input samples)
            for i in range(output_units_count):
                b_i = hidden_to_output_weights[i, h] * hidden_layer_output[:, h]
                output_units_b.append(b_i)
            output_units_b = np.concatenate(output_units_b)  # shape: (1, n_samples * n_output_units)
            b_norms[h] = np.linalg.norm(output_units_b)
        h_star = int(np.argmin(b_norms))

        # mask to keep all hidden units except h_star
        units_to_keep = np.array([i for i in range(hidden_units_count) if i != h_star])

        # least squares system resolution shape (n_samples, hidden_units_count - 1)
        y_no_hidden_unit = hidden_layer_output[:, units_to_keep]

        # we need to update [(hidden_units_count - 1) * output_units_count] weights
        deltas = np.zeros((output_units_count, y_no_hidden_unit.shape[1]))
        for i in range(output_units_count):
            # input for output unit i weighted by the connection (i, h*) over all samples (shape: (n_samples,))
            b_i = hidden_to_output_weights[i, h_star] * hidden_layer_output[:, h_star]

            # compute deltas relative to output unit i's connections (we compute hidden_units_count - 1 deltas)
            delta_i, *_ = np.linalg.lstsq(y_no_hidden_unit, b_i, rcond=None)  # (n_h-1,)
            deltas[i, :] = delta_i

        ''' --- update weights with computed deltas --- '''
        # exclude bias column since it is not directly related to the hidden units
        new_weights = hidden_to_output_weights[:, units_to_keep]
        new_weights[:] += deltas

        new_weights_with_bias = np.hstack([new_weights, hidden_to_output_weights[:,-1].reshape(-1,1)])
        self.w[-1] = new_weights_with_bias

        # remove h_star input connections
        self.w[0] = self.w[0][units_to_keep, :]  # (n_h-1, n_in+1)
        self.hidden_layer_sizes[0] -= 1

        # Resetta lo stato per la momentum (dimensioni cambiate)
        self.previous_update = []
        self.V, self.h, self.d = [], [], []

        return h_star

    def one_hot_encode(self, target: np.ndarray):
        """
        One hot encodes the target vector

        :param target: the output classes for each sample (shape: (n_samples,))
        :return: the one hot encoded matrix (shape: (n_samples, n_classes))
        """
        if self.n_classes == 0:
            self.n_classes = len(np.unique(target))
        encoding = np.eye(self.n_classes)[target]
        return encoding