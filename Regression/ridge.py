import numpy as np
from read_data import read
import matplotlib.pyplot as plt
import pandas as pd


def ridge_regression(traindata_x, traindata_y):
    alpha = 1
    x = traindata_x
    data_length, feature_num = traindata_x.shape
    iterations = 10000

    weight = np.random.rand(feature_num, 1)
    bias = 0
    learning_rate = 0.001
    previous_loss = 1000

    traindata_y = traindata_y.reshape(data_length, 1)
    for i in range(iterations):
        y_pred = np.dot(x, weight) + bias
        y = traindata_y

        loss = np.sum((y - y_pred) ** 2) / data_length + alpha * np.sum(weight ** 2) / (2 * data_length)
        gap = y - y_pred
        partial_weight = -2 / data_length * np.dot(np.transpose(x), gap) + alpha * weight / data_length
        weight -= learning_rate * partial_weight
        partial_bias = np.sum((-2 * gap)) / data_length
        bias -= learning_rate * partial_bias

        if abs(previous_loss - loss) < 1e-8:
            learning_rate = learning_rate * 0.7
        if abs(previous_loss - loss) < 1e-15:
            break
        previous_loss = loss

    final = np.dot(x, weight) + bias
    return weight, bias, final


datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
_, _, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
weight, bias, final = ridge_regression(traindata_x, traindata_y)

with open("C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_ridge_MSE.txt", "w+") as file:
    file.write("Ridge-Variate_norm_weight: ")
    file.write(str(list(np.resize(weight, (weight.shape[0],)))))
    file.write("\nRidge-Variate_norm_bias: ")
    file.write(str(bias))