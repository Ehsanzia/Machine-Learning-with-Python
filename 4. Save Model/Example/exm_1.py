
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("D:/Machine_Learning_with_Python/ML/4_save_model/Example/homeprice.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

import pickle

with open('D:/Machine_Learning_with_Python/ML/4_save_model/Example/model_pickle' , 'wb') as f:
    pickle.dump(reg,f)

with open('D:/Machine_Learning_with_Python/ML/4_save_model/Example/model_pickle' , 'rb') as f:
    mp = pickle.load(f)


import joblib

joblib.dump(mp,'D:/Machine_Learning_with_Python/ML/4_save_model/Example/model_joblib')

mj = joblib.load('D:/Machine_Learning_with_Python/ML/4_save_model/Example/model_joblib')

print(reg.predict([[5000]])) 
print(mp.predict([[5000]]))
print(mj.predict([[5000]]))


