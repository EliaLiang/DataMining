import numpy as np
import pandas as pd
import xlrd
import xlwt
import openpyxl

def normalizing(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    gaps = maxs - mins
    gaps[gaps==0] = 1
    result = (data-mins)/gaps
    return result

#data = pd.read_excel('C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls', 'Sheet1')
#result = normalizing(data)
'''
# create a new file called Normalized_Data.txt and write the result to it
with open('C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.txt', 'w') as file:
    for row in result:
        file.write(' '.join(map(str, row)) + '\n')

#print(result[:10])

'''
# read in your data from the original file
data = pd.read_excel('C:/Users/l.xiao/Desktop/wustl/514/regression/Concrete_Data.xls', 'Sheet1')

# perform normalization on the data
normalized_data = normalizing(data)

# create a new dataframe with the normalized data
df_normalized = pd.DataFrame(normalized_data, columns=data.columns)

# save the new dataframe to a new xlsx file
df_normalized.to_excel('C:/Users/l.xiao/Desktop/wustl/514/regression/Normalized_Data.xls', engine='openpyxl', index=False)
