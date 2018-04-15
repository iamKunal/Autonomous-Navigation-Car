import numpy as np


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self):
        self.layer_sizes = None
        self.layers = None
        self.weights = None
        self.actual_outputs = None
        self.activation_function = sigmoid
        self.targets = None
        self.delta_N = None
        self.learning_rate = 0.01
        self.momentum = 0.0
        self.delta_W = None
        self.epochs = 100
        self.min_accuracy = 0.95

    def set_layer_sizes(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = [0]*len(layer_sizes)
        for i in xrange(len(self.layer_sizes[:-1])):
            print self.layer_sizes[i]
            self.layers[i] = (np.zeros([self.layer_sizes[i]+1], np.float32)) # +1 for bias
        print self.layer_sizes[-1]

        self.layers [-1] = (np.zeros([self.layer_sizes[-1]], np.float32))
        self.weights = []
        for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.rand(i + 1, j) * 2 - 1)  # +1 for bias

    def forward_propagate(self, inputs):
        self.layers[0] = np.append(np.array(inputs), 1)  # Adding bias
        for i in xrange(1, len(self.layer_sizes)-1, 1):

            self.layers[i] = np.append(self.activation_function(self.layers[i - 1].dot(self.weights[i - 1])), 1)

        self.layers[-1] = self.activation_function(self.layers[-2].dot(self.weights[-2]))
    def back_propagate(self, outputs):
        self.targets = np.array(outputs)
        self.delta_N = None
        L = len(self.layer_sizes) - 1
        self.delta_N = [0] * (L + 1)
        self.delta_N[L] = (self.layers[L] - self.targets) * self.activation_function(
            self.weights[L - 1].dot(self.layers[L - 1]), derivative=True)
        for i in xrange(1, L)[::-1]:
            self.delta_N[i] = self.weights[i + 1].T.dot(self.delta_N[i + 1]) * self.activation_function(
                self.weights[i - 1].dot(self.layers[i - 1]))
        self.delta_W = [0] * L
        for i in xrange(L):
            self.delta_W[i] = self.delta_N[i + 1].dot(self.layers[i].T)
            self.weights[i] = self.weights[i] - self.learning_rate * self.delta_W[i]

    def train(self, training_data, training_labels):
        number_of_samples = training_data.shape[0]
        accuracy = 0
        for e in xrange(self.epochs):
            predicted = np.zeros(training_labels.shape[1])
            for i in xrange(number_of_samples):
                self.forward_propagate(training_data[i])
                self.back_propagate(training_labels[i])
                np.vstack((predicted, self.layers[len(self.layer_sizes) - 1]))
            predicted = predicted[1:]
            accuracy = np.mean(predicted == training_labels)
            if accuracy > self.min_accuracy:
                break
        return accuracy

    def predict(self, test_data):
        number_of_samples = test_data.shape[0]
        predicted = np.zeros(test_data.shape[1])
        for i in xrange(number_of_samples):
            self.forward_propagate(test_data[i])
            np.vstack((predicted, self.layers[len(self.layer_sizes) - 1]))
        predicted = predicted[1:]
        return predicted


def get_accuracy(training_labels, predicted):
    return np.mean(predicted == training_labels)
