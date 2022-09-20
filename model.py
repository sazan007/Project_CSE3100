import numpy as np
import pandas as pd
import sklearn.datasets
import pickle
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# loading the data
dataset = sklearn.datasets.load_breast_cancer()


#loading the data to datframe
data_frame = pd.DataFrame(dataset.data,columns=dataset.feature_names)


# adding target columns in the data frame 
data_frame['label']=dataset.target



# Printing the first 5 rows of df...
data_frame.head()




#shape of df
data_frame.shape




#some info of dataset
data_frame.info()



# checking for null values
data_frame.isnull().sum()



#statistical measures about the data
data_frame.describe()




#checking the distribution of target variable
# 1 represents benign and 0 represents malignant
data_frame['label'].value_counts()


# 1-->Benign
# 0-->Malignant


data_frame.groupby('label').mean()




#Separating the dataframe into X and y while X will contain the feature and y will contain label
X= data_frame.drop(columns='label',axis=1)
y= data_frame['label']




print(X)
print(y)




#splitting the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# Using Logistic Regression Model



#train the model
model = LogisticRegression()
model.fit(X_train,y_train)
file_name='model.pkl'
pickle.dump(model,open(file_name,'wb'))


# #Model Evaluation
# #AccuracyTesting
# y_predict = model.predict(X_test)
# accuracy = accuracy_score(y_test,y_predict)
# print('Accuracy on test data = ',accuracy)


# # PREDICTIVE SYSTEM



input_data=(18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368)

# #Change the input data to a numpy array

inp_as_np_array = np.asarray(input_data)

# #reshape the array as we want the prediction for one entry
inp_reshape  = inp_as_np_array.reshape(1,-1)

prediction = model.predict(inp_reshape)

if(prediction[0]==1):
     print('Benign')
else:
     print('Malignant')