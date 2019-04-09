import numpy as np
import random
import pickle
import os

class Network:
    def __init__(self, training_data, layer_sizes, epochs, batch_size, learning_rate):

        self.training_data = training_data
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # first layer doesn't have weights or biases
        self.weights = [ None ]
        self.biases = [ None ]
        self.gradient_weights = [ None ]
        self.gradient_biases = [ None ] 

        # randomly (normal distribution) initialize weights and biases, fill gradient with zeros
        last_size = self.layer_sizes[0]
        for size in self.layer_sizes[1:]:
            self.weights.append(np.random.randn(size, last_size)) 
            self.biases.append(np.random.randn(size))
            self.gradient_weights.append(np.zeros((size, last_size)))
            self.gradient_biases.append(np.zeros(size))
            last_size = size

        self.current_sums = [ None ] * len(self.layer_sizes)
        self.current_activations = [ None ] * len(self.layer_sizes)
        self.current_errors = [ None ] * len(self.layer_sizes)

    def train(self):

        for epoch_counter in range(0, self.epochs):
            random.shuffle(self.training_data)
            print("{} of {} epochs".format(epoch_counter+1, self.epochs))

            right = 0
            for image_counter in range(0, len(self.training_data)):
                self.current_activations[0] = self.training_data[image_counter][0]
                self.current_expected = self.training_data[image_counter][1]

                self.process_next_image()

                predicted = np.argmax(self.current_activations[-1])
                expected = np.argmax(self.current_expected)

                if predicted == expected:
                    right += 1

                if not (image_counter + 1) % self.batch_size:
                    self.flush()

            print("Right: {}%".format(right/len(self.training_data)*100))
            print("")

    def process_next_image(self):
        self.feed_forward()
        self.backpropagation()

    def feed_forward(self):
        for i in range(1, len(self.layer_sizes)):
            self.current_sums[i] = self.weights[i] @ self.current_activations[i - 1] + self.biases[i]
            self.current_activations[i] = Network.sigmoid(self.current_sums[i])

        # Print output layer
        """
        print(self.current_activations[-1])
        print()
        """

    def backpropagation(self):
        self.current_errors[-1] = 1 / self.layer_sizes[-1] * \
            (self.current_activations[-1] - self.current_expected) \
            * Network.sigmoid(self.current_sums[-1])*(1-Network.sigmoid(self.current_sums[-1]))

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            self.current_errors[i] = np.transpose(self.weights[i + 1]) @ self.current_errors[i + 1] \
                * Network.sigmoid(self.current_sums[i])*(1-Network.sigmoid(self.current_sums[i]))

        for i in range(1, len(self.layer_sizes)):
            self.gradient_biases[i] += self.current_errors[i]
            self.gradient_weights[i] += np.tensordot(self.current_errors[i],self.current_activations[i-1],axes=0) 

    def flush(self):
        for i in range(1, len(self.layer_sizes)):
            self.biases[i] -= self.gradient_biases[i] / self.batch_size * self.learning_rate
            self.weights[i] -= self.gradient_weights[i] / self.batch_size * self.learning_rate

        # reset gradients
        self.gradient_biases = [ None ]
        self.gradient_weights = [ None ]
        last_size = self.layer_sizes[0]
        for size in self.layer_sizes[1:]:
            self.gradient_weights.append(np.zeros((size, last_size)))
            self.gradient_biases.append(np.zeros(size))
            last_size = size
    
    def save(self):
        pickle.dump((self.weights, self.biases, self.layer_sizes), open("network.pickle", "wb")) # also save layer_sizes in network file

    def load(self):
        if os.path.isfile("network.pickle"):
            (self.weights, self.biases, self.layer_sizes) = pickle.load(open("network.pickle", "rb"))

    @staticmethod
    def sigmoid(x):
        return 0.5*(1+np.tanh(x/2))