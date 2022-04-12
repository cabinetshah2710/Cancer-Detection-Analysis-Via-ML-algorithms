
#importing libraries

from fileinput import filename
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


# Loading Dataset
prostate=pd.read_csv('C:/Users/Cabinet\Documents/visual studio codes/Model deployment/100 sample prostate cancer.csv')



# Data Transformation
prostate['diagnosis_result'] = prostate['diagnosis_result'].replace(['B'],'0')
prostate['diagnosis_result'] = prostate['diagnosis_result'].replace(['M'],'1')
prostate[['diagnosis_result']] = prostate[['diagnosis_result']].apply(pd.to_numeric, errors ='ignore')

# Splitting Data into Training & Testing Sets
# Defining training and testing data
Y = prostate['diagnosis_result']
X = prostate.drop(columns=['diagnosis_result','id'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)


# Logistic Regression model
# We defining the model
logreg = LogisticRegression(C=10)
# We train the model
logreg.fit(X_train, Y_train)
# We predict target values
Y_predict1 = logreg.predict(X_test)


# Test score - Logistic Regression model
score_logreg = (logreg.score(X_test, Y_test)) * 100
print(score_logreg)
# Logistic Regression model - Precision
average_precision = average_precision_score(Y_test, Y_predict1)
print('Average precision score: {0:0.2f}'.format(average_precision))

# Support Vector classifier model

# We define the SVM model
svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True),n_jobs=-1))
# We train model
svmcla.fit(X_train, Y_train)
# We predict target values
Y_predict2 = svmcla.predict(X_test)



# Support Vector classifier - Test score

score_svmcla = (svmcla.score(X_test, Y_test)) * 100
print(score_svmcla)

# Support Vector classifier - precision

average_precision = average_precision_score(Y_test, Y_predict2)

print('Average precision score: {0:0.2f}'.format(average_precision))

# Naive Bayes model implementation

from sklearn.naive_bayes import GaussianNB

# We define the model
nbcla = GaussianNB()
# We train model
nbcla.fit(X_train, Y_train)
# We predict target values
Y_predict3 = nbcla.predict(X_test)



# Naive Bayes - Test score
score_nbcla = (nbcla.score(X_test, Y_test)) * 100
print(score_nbcla)

# Naive Bayes - precision
average_precision = average_precision_score(Y_test, Y_predict3)

print('Average precision score: {0:0.2f}'.format(average_precision))

# Decision Tree model implementation

# We define the model
dtcla = DecisionTreeClassifier(random_state=9)
# We train model
dtcla.fit(X_train, Y_train)
# We predict target values
Y_predict4 = dtcla.predict(X_test)



#  Decision Tree - Test score

score_dtcla = (dtcla.score(X_test, Y_test)) * 100
print(score_dtcla)

#  Decision Tree - precision

average_precision = average_precision_score(Y_test, Y_predict4)

print('Average precision score: {0:0.2f}'.format(average_precision))

# Random Forest algorithm model implementation



# We define the model
rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)
# We train model
rfcla.fit(X_train, Y_train)
# We predict target values
Y_predict5 = rfcla.predict(X_test)



# Random Forest algorithm - Test score

score_rfcla = (rfcla.score(X_test, Y_test)) * 100
print(score_rfcla)

# Random Forest algorithm - precision


average_precision = average_precision_score(Y_test, Y_predict5)

print('Average precision score: {0:0.2f}'.format(average_precision))

# KNN algorithm model implementation


# We define the model
knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
# We train model
knncla.fit(X_train, Y_train)
# We predict target values
Y_predict6 = knncla.predict(X_test)


# KNN algorithm - Test score
score_knncla= (knncla.score(X_test, Y_test)) * 100
print(score_knncla)

# KNN algorithm - precision 

average_precision = average_precision_score(Y_test, Y_predict6)

print('Average precision score: {0:0.2f}'.format(average_precision))

#Comparison of different models
Testscores = pd.Series([score_logreg , score_svmcla, score_nbcla, score_dtcla, score_rfcla, score_knncla],index=['Logistic Regression Score', 'Support Vector Machine Score', 'Naive Bayes Score', 'Decision Tree Score', 'Random Forest Score', 'K-Nearest Neighbour Score']) 
print(Testscores.sort_values(ascending = False))


width = 0.5
classification = ['L_R', 'S_V_C', 'N_B', 'D_T', 'R_F', 'K-NN']
z = Testscores

plt.bar(classification,z, width, color = ['r','g','b','c','m','y'])
plt.xlabel("Classification", fontsize = 15)
plt.ylabel("Accuracy", fontsize = 15)
plt.title("Comparison of Model", fontsize = 20)

plt.show()


#Model Evaluation of Logistic Regression
print(classification_report(Y_test, Y_predict1))

filename='C:/Users/Cabinet\Documents/visual studio codes/Model deployment/PCP_model.pkl'
joblib.dump(logreg,filename)

