import time

import numpy as np

import reader
import plot_data
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self, X, y, X_cv, y_cv, alpha, ld, layer_size):
        """Init class' value

            alpha: learning rate
            ld: lambda for regularization
            Y: y one-hot
            Y_cv: y_cv one-hot 
            Theta: n2 x n1
            X: m x n
            Y: m x n
        """

        self.alpha = alpha
        self.ld = ld
        self.layer_size = layer_size

        self.X = X
        self.X_cv = X_cv
        self.y = y
        self.y_cv = y_cv

        self.Y = 1. * (np.tile(y, (1, layer_size[-1])) == np.tile(np.arange(layer_size[-1]), (self.X.shape[0], 1)))
        self.Y_cv = 1. * (np.tile(y_cv, (1, layer_size[-1])) == np.tile(np.arange(layer_size[-1]), (self.X_cv.shape[0], 1)))

        self.Theta = []
        for i in range(len(self.layer_size) - 1):
            self.Theta.append(np.random.randn(self.layer_size[i + 1], self.layer_size[i] + 1) * 0.01)

    def get_sigmoid(self, X):
        """Return size(X) matrix"""

        return 1. / (1. + np.exp(-X))

    def get_dSigmoid(self, X):
        """Return size(X) matrix"""

        return self.get_sigmoid(X) * (1. - self.get_sigmoid(X))

    def forward_propagation(self, X):
        """Return list a, z

            a: m x (n + 1)
            z: m x n
        """

        a = [None] * len(self.layer_size)
        z = [None] * len(self.layer_size)
        a[0] = np.insert(X, 0, 1, axis=1)
        for i in range(1, len(self.layer_size)):
            z[i] = np.dot(a[i - 1], self.Theta[i - 1].T)
            a[i] = self.get_sigmoid(np.insert(z[i], 0, 1, axis=1))
        return (a, z)

    def get_cost(self, X, Y, ld, a):
        """Calculate cost function
        
            a: last layer (not contain bias)
            Y: one-hot
        """

        m = X.shape[0]
        return (-1. / m) * np.sum(Y * np.log(a) + (1. - Y) * np.log(1. - a)) \
                + (ld / (2 * m)) * sum(np.sum(x ** 2) for x in self.Theta)

    def predict(self, a):
        """Return m x 1 matrix"""

        return np.reshape(np.argmax(a, axis=1), (a.shape[0], 1))

    def backpropagation(self, a, z):
        """Return gradient
            n2 x n1
        """

        delta = [None] * len(self.Theta)
        sigma = [None] * len(self.layer_size)
        sigma[-1] = a[-1][:, 1:] - self.Y

        for i in range(len(sigma) - 2, 0, -1):
            sigma[i] = np.dot(sigma[i + 1], self.Theta[i][:, 1:]) * self.get_dSigmoid(z[i])
        m = self.X.shape[0]
        for i in range(len(sigma) - 1):
            delta[i] = (1. / m) * np.dot(sigma[i + 1].T, a[i]) + (self.ld / m) * self.Theta[i]
        return delta

    def update(self, delta):
        for i in range(len(self.Theta)):
            self.Theta[i] = self.Theta[i] - self.alpha * delta[i]

    def train(self, iter):
        costs = []
        costs_cv = []
        for i in range(iter):
            a, z = self.forward_propagation(self.X)
            costs.append(self.get_cost(self.X, self.Y, 0, a[-1][:, 1:]))
            delta = self.backpropagation(a, z)
            self.update(delta)
            a, z = self.forward_propagation(self.X_cv)
            costs_cv.append(self.get_cost(self.X_cv, self.Y_cv, 0, a[-1][:, 1:]))
            print(i, costs[-1], costs_cv[-1], self.get_accuraccy(self.X_cv, self.y_cv))
            
        plt.plot(costs, color='blue')
        plt.plot(costs_cv, color='red')
        plt.show()

    def get_accuraccy(self, X, y):
        a, z = self.forward_propagation(X)
        y_predict = self.predict(a[-1][:, 1:])
        return int(sum(y_predict == y)) / y.shape[0]


if __name__ == '__main__':
    start = time.time()

    # read data
    X_train, y_train = reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = reader.load_mnist('data/fashion', kind='t10k')

    # Normalize
    mean = np.mean(X_train)
    std = np.std(X_train)

    X_test = (X_test - mean) / std
    X_train = (X_train - mean) / std

    X_cv = X_test[5000:10000]
    y_cv = y_test[5000:10000]
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:5000]
    y_test = y_test[:5000]

    nn = NeuralNetwork(X_train, y_train, X_cv, y_cv, 0.3, 0, [784, 49, 10])
    nn.train(2000)

    print("test", nn.get_accuraccy(X_test, y_test))
    print("train", nn.get_accuraccy(X_train, y_train))
    print("cv", nn.get_accuraccy(X_cv, y_cv))