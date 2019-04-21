import numpy as np

# Paste your sigmoid function here
def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    # if np.isscalar(z):
    #     s = 0
    # else:
    #     s = np.zeros(z.shape)
    # return s

    s = 1 / (1 + np.exp(-z))
    print(s.shape)

    return s

def convertOneHot(label,n_class):
    # # 1 of K for 10 x 10 
    # a = np.array([1, 0, 3,0,0,0,0,0,0,9])
    # b = np.zeros((10, 10))
    # b[np.arange(10), a] = 1
    # print (b)

    #empty_y

    emptymatrix = np.zeros((len(label),n_class))
    for i in range(len(label)):
        emptymatrix[i,label[i].astype(int)] = 1

    return emptymatrix



# Paste your nnObjFunction here
def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    
    # assune bias at end: 
    # add column of 1s to bias
    #train_data = np.column_stack((train_data, np.ones((train_data.shape[0]))))
    #train_label = np.column_stack((train_label, np.ones((train_label.shape[0]))))



    w1bias = W1[:,n_input]
    w2bias = W2[:,n_hidden]
    w1bias = w1bias.reshape((len(w1bias), 1))
    w2bias = w2bias.reshape((len(w2bias), 1))

    print("biases")
    print(w1bias)
    print(w2bias,"\n")

    W1 = np.delete(W1, -1, axis=1)
    W2 = np.delete(W2, -1, axis=1)

	z = np.matmul(train_data, np.transpose(W1))	


    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    obj_grad = np.zeros(params.shape)


    return (obj_val, obj_grad)

n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)