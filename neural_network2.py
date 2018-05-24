import mnist_reader
from activationFunction import sigmoid, dSigmoid, ReLU, dReLU, softmax
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import math


X_train, y_train = mnist_reader.load_mnist('data/number', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/number', kind='t10k')
X_train = np.transpose(X_train[:1000])
y_train = np.transpose(y_train[:1000])
X_test = np.transpose(X_test[:1000])
y_test = np.transpose(y_test[:1000])
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)


def random_minibatch(X, y, minibatch_size = 64):
    m = X.shape[1]
    mini_batches = []
    y = y.reshape((1, m))
    permutation = list(np.random.permutation(m))
    shuffle_X = X[:, permutation]
    shuffle_Y = y[:, permutation].reshape((1, m))
    num_minibatch = int(math.floor(m / minibatch_size))
    for i in range(num_minibatch):
        mini_batch_X = shuffle_X[:, i * minibatch_size : (i+1) * minibatch_size]
        mini_batch_Y = shuffle_Y[:, i * minibatch_size : (i+1)* minibatch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % minibatch_size != 0:
        mini_batch_X = shuffle_X[:, num_minibatch * minibatch_size : m]
        mini_batch_Y = shuffle_Y[:, num_minibatch * minibatch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


#adam 
def initial_Adam(parameters):
    L = len(parameters) // 2
    V = {}
    S = {}
    for i in range(L):
        V["dW" + str(i+1)] = np.zeros((parameters["W"+ str(i+1)].shape[0], parameters["W" + str(i+1)].shape[1]))
        V["db"+ str(i+1)] = np.zeros((parameters["b"+ str(i+1)].shape[0], parameters["b" + str(i+1)].shape[1]))
        S["dW" + str(i+1)] = np.zeros((parameters["W"+ str(i+1)].shape[0], parameters["W" + str(i+1)].shape[1]))
        S["db"+ str(i+1)] = np.zeros((parameters["b"+ str(i+1)].shape[0], parameters["b" + str(i+1)].shape[1]))
    return V, S


# GRADED FUNCTION: update_parameters_with_adam# GRADED 
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
   
    L = len(parameters) // 2            
    v_corrected = {}
    s_corrected = {}
    
  
    for l in range(L):
        
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)
      
    return parameters, v, s


def initialize_parameters(layer_dims): 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        parameters['b' + str(i)] = np.zeros(shape = (layer_dims[i],1))*0.01
    return parameters


def linear_forward_propagation(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def forward_propagation(A_prev, W,b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward_propagation(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    if activation == "ReLU":
         Z, linear_cache = linear_forward_propagation(A_prev, W, b)
         A, activation_cache = ReLU(Z)
    if activation == "softmax":
        Z, linear_cache = linear_forward_propagation(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_linear_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    caches = []
    for i in range(1, L):
        A_prev = A 
        A, cache = forward_propagation(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], "ReLU")
        caches.append(cache)
       
    AL, cache = forward_propagation(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
   
    return AL, caches


def transform_one_hot(y, num_labels):
   Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (num_labels, len(y))).toarray()
   return Y 


def cost_function(AL, y, lamda, parameters):

   L = len(parameters) // 2
   k = 0
   for i in range(L):
       k += np.sum(parameters["W" + str(i+1)] ** 2)
   Y = transform_one_hot(y, 10)
   m = Y.shape[1]
   cost = -1. / m *(np.sum(np.multiply(np.log(AL), Y))) + lamda / (2 * m) * k
   return cost


def linear_backward(dZ, cache, lamda):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m + lamda / m * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def L_model_backward_propagation(caches, AL, y, lamda):
    L =len(caches)
    grads = {}
    Y = transform_one_hot(y, 10)
    dZL = AL - Y
    linear_cache, activation_cache = caches[L-1]
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward(dZL, linear_cache, lamda)
    L-=1
    while (L > 0):
        linear_cache, activation_cache = caches[L-1]
        dZ = dReLU(grads['dA' + str(L)] , activation_cache)
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward(dZ, linear_cache, lamda)
        L-=1
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grads["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grads["db" + str(i+1)]
    return parameters


def loop(lamda) :
    num_iterations = 100
    costs = []
    t = 0
    parameters = initialize_parameters([X_train.shape[0],128,10])
    # V = initial_velocity(parameters)
    V, S = initial_Adam(parameters)
    for i in range(num_iterations):
        # batch-gradient

        # AL, caches = L_model_linear_forward(X_train, parameters)
        # cost.append(cost_function(AL, y_train, lamda, parameters))
        # if i % 100 == 0:
        #     print(cost_function(AL, y_train, lamda, parameters))
        # # print(AL)
        # grads = L_model_backward_propagation(caches, AL, y_train, lamda)
        # parameters = update_parameters(parameters, grads, 0.25)

        mini_batches =  random_minibatch(X_train, y_train)
        for mini_batch in mini_batches:
            minibatch_X, minibatch_Y = mini_batch
            AL, caches = L_model_linear_forward(minibatch_X, parameters)
            cost = cost_function(AL, minibatch_Y.flatten(), lamda, parameters)
            grads = L_model_backward_propagation(caches, AL, minibatch_Y.flatten(), lamda)
            t = t+1
            parameters , V ,S= update_parameters_with_adam(parameters, grads, V,S,t, 0.0001)
        
        costs.append(cost)
        if i % 1 == 0:
            print ("Cost after iteration  %i: %f" %(i, cost))
    return parameters, grads, costs
    
def getProbsAndPreds(X, parameters):
    probs, cache = L_model_linear_forward(X, parameters)
    preds = np.argmax(probs,axis=0)
    return probs,preds

def getAccuracy(X, Y, parameters):
    prob,preds = getProbsAndPreds(X, parameters)
    accuracy = sum(preds == Y)/(float(len(Y)))
    return accuracy
if (__name__ == "__main__"):
    # mini_batches = random_minibatch(X_train, y_train)
    # X, y = mini_batches[0]
    # print(y.shape, y_train.shape)
    
    # print(transform_one_hot(y.flatten())[:,1])
    
    parameter1s, grads, cost = loop(0.0001)
    train_accuracy = getAccuracy(X_train, y_train, parameter1s)
    print("Train_accuracy: " + str(train_accuracy))
    test_accuracy = getAccuracy(X_test, y_test, parameter1s)
    print("test_accuracy: " + str(test_accuracy))
    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.show()

    


    
    
    
    
   
  