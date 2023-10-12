import numpy as np
import pandas as pd
from math import inf
from read_data import read
import matplotlib.pyplot as plt

#datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
'''
testdata_x = read(datafile)
print(testdata_x[:10])
'''

def MSE(y_pred, y, x, learning_rate, weight, bias):
    data_length = len(y)
    #print(data_length)
    loss = np.sum((y - y_pred) ** 2)/data_length
    #print("Current loss:", loss)  # Add this line to print the loss
    gap = y - y_pred
    partial_weight = np.sum(-2 * gap * x)/data_length
    weight -= learning_rate * partial_weight
    partial_bias = np.sum((-2 * gap)) / data_length
    bias -= learning_rate * partial_bias

    return loss, weight, bias

def MAE(y_pred, y, x, learning_rate, weight, bias):
    data_length  = len(y)
    loss = np.sum(np.abs(y - y_pred)) / data_length
    partial = np.where(y >= y_pred, -1, 1)

    partial_weight = np.sum(partial * x) / data_length
    weight -= learning_rate * partial_weight

    partial_bias = 1.0 / data_length * np.sum(partial)
    bias -= learning_rate * partial_bias

    return loss, weight, bias

def univariant(traindata_x, traindata_y):
    data_length, feature_num = traindata_x.shape
    weight_list = []
    bias_list = []

    for i in range(feature_num):
        x = traindata_x[:, i]
        y = traindata_y
        weight = 0
        bias = 0
        iterations = 10000
        learning_rate = 1e-3
        previous_loss = 1000
        for j in range(iterations):
            y_pred = x * weight + bias
            loss, weight, bias = MSE(y_pred, y, x, learning_rate, weight, bias)
            #loss, weight, bias = MAE(y_pred, y, x, learning_rate, weight, bias)
            if abs(previous_loss - loss) < 1e-8:
                learning_rate = learning_rate * 0.7
            if abs(previous_loss - loss) < 1e-15:
                break
            # print(loss, learning_rate)
            previous_loss = loss

        weight_list.append(weight)
        bias_list.append(bias)
        final = weight * x + bias

        print("all of the feature will be show")
        plt.scatter(traindata_x[:, i], traindata_y)
        plt.title("univariate_MAE_data_normalizing")
        plt.xlabel("x_"+str(i))
        plt.ylabel("y")
        plt.plot(traindata_x[:, i], final, color="red")
        plt.savefig(f"./univariate_MAE{i}.png")
        #plt.clf()
        plt.show()

        print("-------------------------------------------------------------------------------------")
        print(f"Here is feature {i}")
        print("Finish Uni-Variant Linear Regression")
        print("Here is weight : ", weight)
        print("Here is a bias : ", bias)

    return weight_list, bias_list

datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
#datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
weight, bias = univariant(traindata_x, traindata_y)

with open('C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_MAE_uni.txt', 'w') as file:
    file.write("univariate_MAE_weight: ")
    file.write(str(weight))
    file.write("\nunivariate_MAE_bias: ")
    file.write((str(bias)))
    file.close()
# print(weight)
# print(bias)


