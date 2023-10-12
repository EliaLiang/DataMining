import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn import preprocessing


def draw_data():
    #datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls", "Sheet1").to_numpy()
    # minmax = preprocessing.minmax_scale(datafile)
    # std = preprocessing.StandardScaler()
    # datafile = std.fit_transform(datafile)
    datafile = pd.read_excel("C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls", "Sheet1").to_numpy()
    plt.hist(datafile, bins=5, facecolor="green", edgecolor="black")

    #plt.xlabel("Interval")
    #plt.ylabel("Frequency")
    plt.title("Distribution Histograms - Normalizing Data ")
    plt.show()

draw_data()