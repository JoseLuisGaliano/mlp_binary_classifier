import random
import numpy as np


class MLP:
    weights = []
    biases = []

    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.normal(0, 0.01, (layers[i], layers[i - 1])))
            self.biases.append(np.random.normal(0, 0.01, (layers[i], 1)))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def derivative_of_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feed_forward(self, sample):
        linear_outputs = []
        activated_outputs = []
        input = np.reshape(sample, (4, 1))
        activated_outputs.append(input)  # necessary for the last backpropagation step
        for layer in range(len(self.weights)):
            layer_output = np.dot(self.weights[layer], input) + self.biases[layer]
            activated_output = self.sigmoid(layer_output)
            input = activated_output
            linear_outputs.append(layer_output)
            activated_outputs.append(activated_output)
        return linear_outputs, activated_outputs

    def backpropagate(self, label, linear_output, activated_output):
        dW = []
        db = []
        output_error = activated_output[len(activated_output) - 1] - label
        db.append(output_error)
        dW.append(np.dot(output_error, np.transpose(activated_output[len(activated_output) - 2])))
        next_layer_error = output_error
        for layer in reversed(range(len(self.weights) - 1)):
            spz = self.derivative_of_sigmoid(linear_output[layer])
            next_layer_error = np.dot(np.transpose(self.weights[layer + 1]), next_layer_error) * spz
            db.append(next_layer_error)
            dW.append(np.dot(next_layer_error, np.transpose(activated_output[layer])))
        return dW, db

    def update_parameters(self, gradients, learning_rate):
        dW = gradients[0]
        db = gradients[1]
        grad_index = len(dW) - 1  # necessary since weights go from input to output and gradients from output to input
        for layer in range(len(self.weights)):
            self.weights[layer] = self.weights[layer] - learning_rate * dW[grad_index]
            self.biases[layer] = self.biases[layer] - learning_rate * db[grad_index]
            grad_index = grad_index - 1

    def classify(self, x):
        input = np.reshape(x, (4, 1))
        for layer in range(len(self.weights)):
            layer_output = np.dot(self.weights[layer], input) + self.biases[layer]
            activated_output = self.sigmoid(layer_output)
            input = activated_output
        return activated_output

    def compute_loss(self, X, Y):
        n = len(Y)
        A = []
        for sample in X:
            A.append(self.classify(sample))
        A = np.squeeze(A)
        epsilon = 1e-15  # small constant to avoid log(0)
        A = np.clip(A, epsilon, 1 - epsilon)  # clip values to avoid log(0) and log(1)
        return np.mean(-(1 / n) * (Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    def train(self, samples, labels, epochs, learning_rate, cost_progression=0):
        for epoch in range(epochs):
            for sample, label in zip(samples, labels):
                linear_output, activated_output = self.feed_forward(sample)
                gradients = self.backpropagate(label, linear_output, activated_output)
                self.update_parameters(gradients, learning_rate)
            if cost_progression == 1:
                cost = self.compute_loss(samples, labels)
                print("Cost value in Epoch " + str(epoch) + ": " + str(cost))
