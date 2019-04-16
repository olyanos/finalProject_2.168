import pandas as pd

baseDirectory = '../'
dataDirectory = baseDirectory + "Data/"

f1 = pd.read_csv(dataDirectory + 'CompleteData_Upod1.csv')
f2 = pd.read_csv(dataDirectory + 'CompleteData_Upod2.csv')

print(list(f1))

Xchannels = ['Fig1r', 'Fig2r', 'Temp', 'RH']
Ychannels = ['Acet', 'Benz', 'Form', 'Meth', 'Tol']

X = f1[Xchannels]
Y = f1[Ychannels]

print(X.shape, Y.shape)