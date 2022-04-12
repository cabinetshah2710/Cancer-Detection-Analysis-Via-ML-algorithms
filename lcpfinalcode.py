import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import joblib
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Data Import
train_df=pd.read_csv('C:/Users/Cabinet/Documents/visual studio codes/Model deployment/survey lung cancer.csv')

  

train_df["GENDER"] = train_df["GENDER"].replace(['F'],'0')
train_df["GENDER"] = train_df["GENDER"].replace(['M'],'1')
train_df[["GENDER"]] = train_df[["GENDER"]].apply(pd.to_numeric, errors ='ignore')
train_df["LUNG_CANCER"]= train_df["LUNG_CANCER"].replace(["NO"],'0')
train_df["LUNG_CANCER"]= train_df["LUNG_CANCER"].replace(["YES"],'0')
train_df[["LUNG_CANCER"]] = train_df[["LUNG_CANCER"]].apply(pd.to_numeric, errors ='ignore')


#splitting Training and testing data
X = train_df.drop(columns=["LUNG_CANCER"])
y = train_df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42 ) #test-train data split - 20/80

X_train.shape
X_test.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



logreg = LogisticRegression(C=10)
logreg.fit(X_train, y_train)
Y_predict1 = logreg.predict(X_test)


filename='C:/Users/Cabinet/Documents/visual studio codes/Model deployment/LCP_model.pkl'
joblib.dump(logreg,filename)