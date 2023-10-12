import numpy as np
import pandas as pd
from sklearn import preprocessing


def read(datafile):
    #datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
    #datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
    np.random.shuffle(datafile)
    #  split dataset to train / test
    traindata = datafile[:900, :]
    traindata_y = traindata[:, -1]
    traindata_x = traindata[:, :-1]

    testdata = datafile[900:, :]
    testdata_y = testdata[:, -1]
    testdata_x = testdata[:, :-1]

    data_y = datafile[:, -1]
    data_x = datafile[:, :-1]

    return data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y

#read()

    #return testdata_x
'''
datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
testdata_x = read(datafile)
print(testdata_x[:10])

    #read()
'''
