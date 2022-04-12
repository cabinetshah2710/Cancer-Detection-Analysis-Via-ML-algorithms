import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r'C:/Users/Cabinet/Documents/visual studio codes/Model deployment/weight-height.csv', header=0 , index_col=None)
#print(data.head())

x = data[['Height']]
#print(x.head())
y = data[['Weight']]
#print(y.head())

# spliting train & test data 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)

# creating instance of LinearRegression 
lr = LinearRegression()

# fitting train set of data into model
lr.fit(x_train,y_train)

# predicting values on x_test set of data
y_pred = lr.predict(x_test)

# MSE 
mse = mean_squared_error(y_pred,y_test)

print("MSE :" ,round(mse,2))

filedata = 'C:/Users/Cabinet/Documents/visual studio codes/Model deployment/lr_model.pkl'
joblib.dump(lr,filedata)

