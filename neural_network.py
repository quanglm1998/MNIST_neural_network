import time

import numpy as np

import reader
import plot_data


class NeuralNetwork(object):
    """A neural network takes training test set to optimize theta"""

    def __init__(self, X, y, alpha=0.3, ld=0, layer_size=[]):
        """Init neural network

        Arg:
            X: training examples m x n with m - number of traing tests, n - number of attributes
            y: training results m x 1, haven't one hot transformed yet
            alpha: learning rate, default: 0.3
            ld: lambda for regularization, default: 0
            layer_size: number of units in each layer without bias
            m: number of tests
        """

        self.alpha = alpha
        self.ld = ld
        self.m = X.shape[0]
        self.layer_size = layer_size

        self.X = X
        self.X = self.X / 256   # normalize data
        self.Y = 1. * (np.tile(y, (1, layer_size[-1])) == np.tile(np.arange(layer_size[-1]), (self.m, 1)))
        self.Theta = []
        for i in range(len(self.layer_size) - 1):
            self.Theta.append(np.random.rand(self.layer_size[i + 1], self.layer_size[i] + 1) * 0.01)

    @classmethod
    def get_sigmoid(self, X):
        """Return size(X) matrix"""

        return 1. / (1. + np.exp(-X))

    @classmethod
    def get_dSigmoid(self, X):
        """Return size(X) matrix"""

        return self.get_sigmoid(X) * (1. - self.get_sigmoid(X))

    def get_cost_and_grad(self, X, Y, getGrad=False, ld=0., predict=False):
        """Return cost function and gradient if set and predict

            a: m x (n + 1)
            z: m x n
            sigma: m x n
        """

        m = X.shape[0]
        
        # Feedforward
        a = [None] * len(self.layer_size)
        z = [None] * len(self.layer_size)
        a[0] = np.insert(X, 0, 1, axis=1)
        for i in range(1, len(self.layer_size)):
            z[i] = np.dot(a[i - 1], self.Theta[i - 1].T)
            a[i] = self.get_sigmoid(np.insert(z[i], 0, 1, axis=1))
        a[-1] = a[-1][:, 1:]

        if predict:
            return np.reshape(np.argmax(a[-1], axis=1), (a[-1].shape[0], 1))
 
        # Cost function
        cost = (-1. / m) * np.sum(Y * np.log(a[-1]) + (1. - Y) * np.log(1. - a[-1])) \
                + (ld / (2 * m)) * sum(np.sum(x ** 2) for x in self.Theta)

        if not getGrad:
            return cost

        # Backpropagation
        delta = [None] * len(self.Theta)
        sigma = [None] * len(self.layer_size)
        sigma[-1] = a[-1] - Y

        for i in range(len(sigma) - 2, 0, -1):
            sigma[i] = np.dot(sigma[i + 1], self.Theta[i][:, 1:]) * self.get_dSigmoid(z[i])
        for i in range(len(sigma) - 1):
            delta[i] = (1. / m) * np.dot(sigma[i + 1].T, a[i]) + (ld / m) * self.Theta[i]

        return (cost, delta)

    def train(self, iter):
        costs = []
        for i in range(iter):
            (cost, grad) = self.get_cost_and_grad(self.X, self.Y, getGrad=True, ld=self.ld)
            for j in range(len(self.Theta)):
                self.Theta[j] = self.Theta[j] - self.alpha * grad[j]
            if i > 8000:
                costs.append(cost)
            if i % 100 == 0:
                print(i, cost)

        plot_data.show_list(costs)

    def predict(self, X):
        res = self.get_cost_and_grad(X, nn.Y, predict=True)
        return np.reshape(res, (res.shape[0], 1))

    def gradient_checking(self, eps=1e-4):
        (cost, grad) = self.get_cost_and_grad(self.X, self.Y, getGrad=True, ld=self.ld)
        num_grad = [None] * len(grad)
        for i in range(len(grad)):
            print(i)
            num_grad[i] = np.zeros(grad[i].shape)
            for j in range(grad[i].shape[0]):
                for k in range(grad[i].shape[1]):
                    self.Theta[i][j][k] -= eps
                    cost1 = self.get_cost_and_grad(self.X, self.Y)
                    self.Theta[i][j][k] += eps * 2.
                    cost2 = self.get_cost_and_grad(self.X, self.Y)
                    self.Theta[i][j][k] -= eps
                    num_grad[i][j][k] = (cost2 - cost1) / (2 * eps)
        for i in range(len(grad)):
            print(np.max(np.abs(num_grad[i] - grad[i])))


if __name__ == '__main__':
    start = time.time()

    # read data
    X_train, y_train = reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = reader.load_mnist('data/fashion', kind='t10k')

    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    nn = NeuralNetwork(X_train, y_train, alpha=0.3, layer_size=[784, 10])
    nn.train(1500)

    y_predict = nn.predict(X_test)
    passed = sum(y_predict == y_test)
    print("Passed: %d" % passed)
    print("Total : %d" % y_predict.shape[0])
    print("Running time %0.4f s" % (time.time() - start))
    