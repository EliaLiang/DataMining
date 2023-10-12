import numpy as np
from read_data import read
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(913)

def multi_variant(traindata_x, traindata_y):
    x = traindata_x
    data_length, feature_num = traindata_x.shape
    iterations = 10000

    weight = np.random.rand(feature_num, 1)
    bias = 0
    learning_rate = 0.001
    previous_loss = 1000

    traindata_y = traindata_y.reshape(900, 1)
    for j in range(iterations):
        y_pred = np.dot(x, weight)+bias
        y = traindata_y
        '''#MSE
        loss = np.sum((y - y_pred) ** 2)/data_length
        # print(loss)
        gap = y - y_pred
        # print(x.shape, gap.shape)
        partial_weight = -2/data_length * np.dot(np.transpose(x), gap)
        weight -= learning_rate * partial_weight
        partial_bias = np.sum((-2 * gap)) / data_length
        bias -= learning_rate * partial_bias
        #print(learning_rate)'''
        #MAE
        loss = np.sum(np.abs(y - y_pred)) / data_length
        gap = np.where(y >= y_pred, -1, 1)
        partial_weight = np.dot(np.transpose(x), gap)
        #partial_weight = np.sum(gap * x) / data_length
        weight -= learning_rate * partial_weight
        partial_bias = np.sum(gap) / data_length
        bias -= learning_rate * partial_bias

        if abs(previous_loss - loss) < 1e-4:
            learning_rate = learning_rate * 0.7
        previous_loss = loss
    final = np.dot(x, weight)+bias
    return weight, bias, final

datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
_, _, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
weight, bias, final = multi_variant(traindata_x, traindata_y)

with open("C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_Multi_MAE.txt", "w+") as file:
    file.write("Multi-Variate_norm_weight: ")
    file.write(str(list(np.resize(weight, (weight.shape[0],)))))
    file.write("\nMulti-Variate_norm_bias: ")
    file.write(str(bias))
