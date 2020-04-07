"""
@shilpath
An MLP involving both forward prop and back prop implemented from scratch using NUMPY.
It is trained and tested on the MNIST data set.
Includes tanh and ReLu activation functions.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def relu_func(op_layer):
    op_layer[op_layer<0] = 0
    return(op_layer)


def tanh_func(op_layer):
    return np.tanh(op_layer)


def softmax_func(num_vec):
    denominator = sum(np.exp(num_vec))
    op_list = [np.exp(num_vec[i])/denominator for i in range(len(num_vec))]
    return op_list


def MLP_classifier(x_data, w_list, b_list):
    op_layer1 = (x_data@np.transpose(w_list[0]) + np.transpose(b_list[0]))
    op_layer1 = relu_func(op_layer1)
    '''
    For tanh activation function comment above line and uncomment below line
    '''
    #op_layer1 = tanh_func(op_layer1)

    op_layer2 = (op_layer1@np.transpose(w_list[1]) + np.transpose(b_list[1]))
    op_layer2 = relu_func(op_layer2)
    '''
    For tanh activation function comment above line and uncomment below line
    '''
    #op_layer2 = tanh_func(op_layer2)

    op_layer3 = (op_layer2@np.transpose(w_list[2]) + np.transpose(b_list[2]))
    for i in range(op_layer3.shape[0]):
        denominator = sum(np.exp(op_layer3[i]))
        op_layer3[i] = [np.exp(op_layer3[i][j])/denominator for j in range(len(op_layer3[i]))]
        max_index = np.argmax(op_layer3[i])
        op_layer3[i] = [0]*len(op_layer3[i])
        op_layer3[i][max_index] = 1

    return (op_layer1, op_layer2, op_layer3)


def derivative_relu(a):
    a[a<=0] = 0
    a[a>0] = 1
    return a


def derivative_tanh(a):
    retmat = 1 - (np.tanh(a))**2
    return retmat


def delta_softmax(y_pred, y_actual):
    return y_pred - y_actual


def delta_hidden(a_n, w_n_plus_1, delta_n_plus_1):
    a_derivative_n = derivative_relu(a_n)
    '''
    For tanh activation function comment above line and uncomment below line
    '''
    #a_derivative_n = derivative_tanh(a_n)
    prod = delta_n_plus_1 @ w_n_plus_1
    delta_hid = np.multiply(a_derivative_n, prod)
    return delta_hid


def update_weights(learning_rate, w_n, delta_n, a_n_minus_1, batch_size):
    w_n_updated = w_n - (learning_rate*(np.transpose(delta_n) @ a_n_minus_1))/batch_size
    return w_n_updated


def update_biases(learning_rate, b_n, delta_n, batch_size):
    a_temp = np.ones((delta_n.shape[0], 1))
    b_n_updated = b_n - (learning_rate*(np.transpose(delta_n) @ a_temp))/batch_size
    return b_n_updated


if __name__ == '__main__':

    DATA_FNAME = 'mnist_traindata.hdf5'
    key_list = []
    df = pd.DataFrame()
    with h5py.File(DATA_FNAME, 'r') as hf:
        xdata = hf['xdata'][:]
        ydata = hf['ydata'][:]

    np.random.seed(5)

    indices = np.random.permutation(len(xdata))
    xdata = xdata[indices]
    ydata = ydata[indices]
    x_train = xdata[:50000, :]
    y_train = ydata[:50000, :]
    x_val = xdata[50000:, :]
    y_val = ydata[50000:, :]

    learning_rate = 0.005  #learning rate
    bs = 100  #batch size
    num_epochs = 5  #number of epochs
    w1 = np.random.randn(200, xdata.shape[1])*0.01
    w2 = np.random.randn(100, 200)*0.01
    w3 = np.random.randn(10, 100)*0.01
    b1 = np.random.randn(200, 1)
    b2 = np.random.randn(100, 1)
    b3 = np.random.randn(10,1)
    w_list = [w1, w2, w3]
    b_list = [b1, b2, b3]
    val_accuracy_list = []
    train_accuracy_list = []

    for i in range(num_epochs):
        count_train_misclassified = 0
        count_val_misclassified = 0
        if i==20:
            learning_rate = learning_rate/2
        if i==40:
            learning_rate = learning_rate/2
        print("training epoch ", i, " of ", num_epochs)
        for j in range(0, 50000, bs):
            a1, a2, a3 = MLP_classifier(x_train[j:j+bs,:], w_list, b_list)
            del_softmax_l3 = delta_softmax(a3, y_train[j:j+bs,:])
            del_hidden_l2 = delta_hidden(a2, w3, del_softmax_l3)
            del_hidden_l1 = delta_hidden(a1, w2, del_hidden_l2)

            w3 = update_weights(learning_rate, w3, del_softmax_l3, a2, bs)
            w2 = update_weights(learning_rate, w2, del_hidden_l2, a1, bs)
            w1 = update_weights(learning_rate, w1, del_hidden_l1, x_train[j:j+bs,:], bs)

            b3 = update_biases(learning_rate, b3, del_softmax_l3, bs)
            b2 = update_biases(learning_rate, b2, del_hidden_l2, bs)
            b1 = update_biases(learning_rate, b1, del_hidden_l1, bs)
            w_list = [w1, w2, w3]
            b_list = [b1, b2, b3]

        #calculate train accuracy
        a1, a2, a3 = MLP_classifier(x_train, w_list, b_list)
        diff = a3 - y_train
        for i in range(len(diff)):
            if 1 in diff[i]:
                count_train_misclassified += 1

        #calculate val accuracy
        v1, v2, v3 = MLP_classifier(x_val, w_list, b_list)
        diff2 = v3 - y_val
        for i in range(len(diff2)):
            if 1 in diff2[i]:
                count_val_misclassified += 1

        count_train_correct = x_train.shape[0] - count_train_misclassified
        train_accuracy = count_train_correct/x_train.shape[0]
        train_accuracy_list.append(train_accuracy)

        count_val_correct = x_val.shape[0] - count_val_misclassified
        val_accuracy = count_val_correct/x_val.shape[0]
        val_accuracy_list.append(val_accuracy)

        print("Training accuracy = ", train_accuracy, "    Validation accuracy = ", val_accuracy)

        #shuffle training data at the end of each epoch
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

    epochs = np.arange(num_epochs)
    plt.figure()
    plt.plot(epochs, train_accuracy_list, label='train_accuracy')
    plt.plot(epochs, val_accuracy_list, label='val_accuracy')
    plt.axvline(x=20)
    plt.axvline(x=40)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Backpropagation ReLu activation LR=0.005 ')
    plt.legend()
    plt.savefig(f'Backpropagation_lr_0.005_relu.png', dpi=256)
