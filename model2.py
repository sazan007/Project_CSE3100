import pandas as pd
import numpy as np
#import matplotlib.pyplot as mp
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('E:\\3100\\css\\nav\\nav2\\Dataset\\parkinsons.csv')

x = df.drop(['name','status'],axis=1)
y = df['status']

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
x_ros,y_ros = ros.fit_resample(x,y)

#scaling the data 
scaler = MinMaxScaler((-1,1))
#x= scaler.fit_transform(x_ros)
x=x_ros
y=y_ros

x_pca=x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.3,random_state=7)
x_pca.shape


#fitting model

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(x_train,y_train)

import pickle

file_name='model2.pkl'
pickle.dump(model,open(file_name,'wb'))


