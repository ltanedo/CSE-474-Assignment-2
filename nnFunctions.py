import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

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

    divider = (1 + np.exp(-z))
    s = 1 / divider
    return s
        
def convertOneHot(label,n_class):

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

    W1_orig = W1.copy()
    W2_orig = W2.copy()

    train_data_w_ones = np.column_stack((train_data, np.ones((train_data.shape[0]))))

    ########################## feed foward propogation ################################

    n = np.shape(train_data)[0]

    W1 = np.delete(W1, -1, axis=1)
    W2 = np.delete(W2, -1, axis=1)

    print("w1 no bias")
    print(W1,"\n")

    print("w2 no bias")
    print(W2,"\n")

    #z = sigmoid(np.add(np.dot(train_data,np.transpose(W1)) , w1bias))
    z = sigmoid((np.dot(train_data_w_ones,np.transpose(W1_orig))))
    z = np.column_stack((z, np.ones((z.shape[0]))))
    #o = sigmoid(np.add(np.dot(z,np.transpose(W2)) , w2bias))
    o = sigmoid(np.dot(z,np.transpose(W2_orig)))
    print("o")
    print(o, "\n")
    print("z")
    print(z, "\n")
    
    ################################### JW Error ###################################

    # y -> oneHot
    y = convertOneHot(train_label,n_class)
    #print("\nonehot y")
    #print(y, "\n")
    # print(n)
    third = np.log(1 - o)
    # print (third)
    second = 1 - y
    first = y*np.log(o)
    last = first + second*third
    obj_val = np.sum(last) * -(1/train_data.shape[0] )



    ########################### W2 Update ############################

    w2_return = np.matmul(np.transpose(o - y) , z)
    

    ########################### W1 Update ############################

    #all z needs biases
    #outer product(np.transpose(x))
    #strip bottomrow end

    left = (1 - z) * z

    # print(left.shape)
    
    delta = o - y
    middle = np.dot(delta, W2_orig)

    #strip middle
    # middle = np.delete(middle, -1, axis=1)

    grad_W1 = np.dot(np.transpose(left * middle),train_data_w_ones)
    grad_W1 = np.delete(grad_W1, n_hidden, 0)

    #print(grad_W1)
    #result
    #grad_W1 = np.outer(np.matmul(left, middle), training_data)

    #print("delta")
    #print(delta,"\n")

    # print("\nSUM CHECK\n","delta: ",delta.shape,"\nW2.shape: ",W2.shape,)
    # vec = np.zeros((2,3))
    # for i in range(2):
    #     vec = delta[i] * W2[i]

    # print(middle.shape)

    #preRight = np.matmul(left,middle)
    # print(preRight.shape)
    # print (train_data.shape, "\n")
    
    #w1_return = np.matmul(np.transpose(preRight), train_data)
    # print(w1_return.shape, "\n")

    #grad_W1 = w1_return
    grad_W2 = w2_return

    # print("w1 and w2 shapes")
    # print(w2_return.shape,"\n")

    # print("grad_W1")
    # print(grad_W1,"\n")
    # print("grad_W2")
    # print(grad_W2,"\n")

    ############################ REG OBJ VAL ##############################

    #W1_square = np.mat(np.transpose(W1_orig),W1_orig)
    # #W1_square = np.square(w1_return)
    W1_square = W1_orig * W1_orig
    W1_square_sum = np.sum(W1_square)

    #W2_square = np.matmul(np.transpose(W2_orig),W2_orig)
    # #W2_square = np.square(w2_return)
    W2_square = W2_orig * W2_orig
    W2_square_sum = np.sum(W2_square)

    inner_sum = W1_square_sum + W2_square_sum

    regTerm = lambdaval / (2 * n)
    # #print(regTerm)

    rightSide = regTerm * inner_sum

    # print("obj_val -> regularized")
    obj_val = obj_val + rightSide
    # print(obj_val,"\n")

    ############################ W2 Reg ##############################

    # W2 -> Post Reg
    # print(w2_return.shape)
    # print(W2.shape)
    # print(grad_W2)
    W2_reg = (1 / n) * np.add(grad_W2, (lambdaval * W2_orig))
    print("W2 -> regularized")
    print(W2_reg,"\n")

    grad_W2 = W2_reg

    ############################ W2 Reg ##############################

    # # w1 -> Post Reg
    W1_reg = (1 / n) * np.add(grad_W1, (lambdaval * W1_orig))
    # print("W1 -> regularized")
    # print(W1_reg,"\n")

    grad_W1 = W1_reg

    ############################ Output #######################################

    #outputs

        # obj_val: a scalar value representing value of error function

        # obj_grad: a SINGLE vector (not a matrix) of gradient value of error function


    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # print(grad_W1.shape)
    # print(grad_W2.shape)

    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    #obj_grad = np.zeros(params.shape)
    # print(len(obj_grad))
    return (obj_val, obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''

    labels = np.zeros((data.shape[0],))
    # Your code here

    W1_orig = W1.copy()
    W2_orig = W2.copy()
    train_data = data.copy()
    train_data_w_ones = np.column_stack((train_data, np.ones((train_data.shape[0]))))

    n_input = train_data.shape[1]
    n_hidden = W1_orig.shape[0]
    n_class = W2_orig.shape[0]

    print("W1 w bias")
    print(W1,"\n")
    print("W2 w bias")
    print(W2,"\n")


    ########################## feed foward propogation ################################

    n = np.shape(train_data)[0]

    w1bias = W1[:,n_input]
    w2bias = W2[:,n_hidden]
    # w1bias_with_one = w1bias.reshape((len(w1bias), 1))
    # w2bias_with_one = w2bias.reshape((len(w2bias), 1))

    #print(train_data)

    print("biases")
    print(w1bias)
    print(w2bias,"\n")

    w1bias = np.tile(w1bias,(n_class,1))
    w2bias = np.tile(w2bias,(n_class,1))

    #w1bias = np.repeat(w1bias, 2, axis=0)
    #w2bias = np.repeat(w2bias, 3, axis=1)


    print("biases to match z and o")
    print(w1bias)
    print(w2bias,"\n")

    W1 = np.delete(W1, -1, axis=1)
    W2 = np.delete(W2, -1, axis=1)

    print("w1 no bias")
    print(W1,"\n")

    print("w2 no bias")
    print(W2,"\n")

    #print(np.matmul(train_data,np.transpose(W1)))

    #z = sigmoid(np.add(np.dot(train_data,np.transpose(W1)) , w1bias))
    z = sigmoid((np.dot(train_data_w_ones,np.transpose(W1_orig))))
    z = np.column_stack((z, np.ones((z.shape[0]))))

    print("z")
    print(z, "\n")

    # print("z_w/bias")
    # print(z_with_bias, "\n")

    #o = sigmoid(np.add(np.dot(z,np.transpose(W2)) , w2bias))
    o = sigmoid(np.dot(z,np.transpose(W2_orig)))
    print("o")
    print(o, "\n")

    labels = o

    print(type(labels))

    return labels
