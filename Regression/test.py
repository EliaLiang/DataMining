import numpy as np
from read_data import read
import os
import pandas as pd


def var_univariant(x, y):
    data_length, feature_num = x.shape
    filename = 'C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_MAE_uni.txt'
    r2_lst = []
    with open(filename) as f:
        file = f.read()
        weights = np.array([float(i) for i in (
            list(list(file.split('\n'))[0].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])
        bias = np.array([float(i) for i in (
            list(list(file.split('\n'))[1].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])

    for i in range(feature_num):
        y_pred = x[:, i] * weights[i] + bias[i]

        #mse = np.sum((y - y_pred) ** 2) / data_length
        mae = np.sum(np.abs(y - y_pred)) / data_length
        var = np.sum((y - y.mean()) ** 2) / data_length

        r2 = 1 - mae / var
        r2_lst.append(r2)
    return r2_lst


def var_multivariant(x, y):
    data_length, feature_num = x.shape
    filename = 'C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_Multi_MAE.txt'

    with open(filename) as f:
        file = f.read()
        weight = np.array([float(i) for i in (
            list(list(file.split('\n'))[0].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])
        bias = np.array([float(i) for i in (
            list(list(file.split('\n'))[1].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])[0]

    y_pred = np.dot(x, weight)+bias

    #mse = np.sum((y - y_pred) ** 2) / data_length
    mae = np.sum(np.abs(y - y_pred)) / data_length
    var = np.sum((y - y.mean()) ** 2) / data_length
    r2 = 1 - mae / var
    return r2

def var_ridge(x, y):
    data_length, feature_num = x.shape
    filename = 'C:/Users/l.xiao/Desktop/wustl/514/regression/norm_data/hyperparameter_ridge_MSE.txt'

    with open(filename) as f:
        file = f.read()
        weight = np.array([float(i) for i in (
            list(list(file.split('\n'))[0].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])
        bias = np.array([float(i) for i in (
            list(list(file.split('\n'))[1].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])[0]

    y_pred = np.dot(x, weight) + bias

    mse = np.sum((y - y_pred) ** 2) / data_length
    #mae = np.sum(np.abs(y - y_pred)) / data_length
    var = np.sum((y - y.mean()) ** 2) / data_length
    r2 = 1 - mse / var
    return r2

def train_raw():
    datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
    data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
    r2_uni = var_univariant(traindata_x, traindata_y)
    r2_multi = var_multivariant(traindata_x, traindata_y)
    print("train data R-squared on raw data:")
    r2_uni.append(r2_multi)
    print(r2_uni)

def train_preprocessed():
    datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
    data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
    #r2_uni = var_univariant(traindata_x, traindata_y)
    #r2_multi = var_multivariant(traindata_x, traindata_y)
    r2_ridge = var_ridge(traindata_x, traindata_y)
    print("train data R-squared on pre-processed data:")
    #r2_uni.append(r2_multi)
    #print(r2_uni)
    print(r2_ridge)

def test_raw():
    datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
    data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
    r2_uni = var_univariant(testdata_x, testdata_y)
    r2_multi = var_multivariant(testdata_x, testdata_y)
    print("test data R-squared on raw data:")
    r2_uni.append(r2_multi)
    print(r2_uni)

def test_preprocessed():
    datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
    data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read(datafile)
    #r2_uni = var_univariant(testdata_x, testdata_y)
    #r2_multi = var_multivariant(testdata_x, testdata_y)
    r2_ridge = var_ridge(traindata_x, traindata_y)
    print("test data R-squared on pre-processed data:")
    #r2_uni.append(r2_multi)
    #print(r2_uni)
    print(r2_ridge)

#train_raw()
train_preprocessed()
#test_raw()
test_preprocessed()