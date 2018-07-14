import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('diabetes.csv')

dataset.describe()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,[8]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 0:6])
X[:, 0:6] = imputer.transform(X[:, 0:6])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''NAIVE BAYES'''

from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)

accuracy_naivebayes = (cm1[0][0]+cm1[1][1])/154





''' SVC'''

from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf', random_state = 0)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y_pred2)
#accuracy_kernelSVC = (cm2[0][0]+cm2[1][1])/154

''' applying parameter tununig'''

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,2,4,6], 'kernel': ['linear']},
              {'C': [1,2,4,6], 'kernel': ['rbf'], 'gamma': [0.125, 0.2, 0.1,0.25]}]
grid_search = GridSearchCV(estimator = classifier2,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
                         
grid_search = grid_search.fit(X_train, y_train)
best_accuracy_KSVC = grid_search.best_score_
best_parameters_KSVC = grid_search.best_params_

''' we obtain c=1,kernel=linear'''

classifier22 = SVC(kernel = 'linear', random_state = 0)
classifier22.fit(X_train, y_train)

# Predicting the Test set results
y_pred22 = classifier22.predict(X_test)

from sklearn.metrics import confusion_matrix
cm22=confusion_matrix(y_test,y_pred22)
accuracy_kernelSVC = (cm22[0][0]+cm22[1][1])/154

''' after applying parameter tuning accuracy has increased '''


'''RANDOMFOREST '''


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators =100, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)

from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(y_test,y_pred3)


accuracy_forest = (cm3[0][0]+cm3[1][1])/154


from sklearn.model_selection import GridSearchCV
parameters3 = [{'n_estimators': [50,100,150,200]}]
             
grid_search3 = GridSearchCV(estimator = classifier3,
                           param_grid = parameters3,
                           scoring = 'accuracy',
                           cv = 10)
                         
grid_search3 = grid_search3.fit(X_train, y_train)
best_accuracy_forest = grid_search3.best_score_
best_parameters_forest = grid_search3.best_params_

''' we found that the best n_estimators=100'''


''' K NEIGHBORS '''


from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 25, metric = 'minkowski', p = 2)
classifier4.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier4.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)

accuracy_kneighbors = (cm4[0][0]+cm4[1][1])/154

from sklearn.model_selection import GridSearchCV
parameters4= [{'n_neighbors': [10,20,25,30]}]
             
grid_search4 = GridSearchCV(estimator = classifier4,
                           param_grid = parameters4,
                           scoring = 'accuracy',
                           cv = 10)
                         
grid_search4 = grid_search4.fit(X_train, y_train)
best_accuracy_KNN = grid_search4.best_score_
best_parameters_KNN = grid_search4.best_params_



'''LOGISTIC REGRESSION'''

from sklearn.linear_model import LogisticRegression

classifier5=  LogisticRegression(C=0.3)
classifier5.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier5.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred5)
accuracy_Lregression= (cm5[0][0]+cm5[1][1])/154

from sklearn.model_selection import GridSearchCV
parameters5= [{'C': [0.5,0.7,0.3,1,0.1]}]
             
grid_search5 = GridSearchCV(estimator = classifier5,
                           param_grid = parameters5,
                           scoring = 'accuracy',
                           cv = 10)
                         
grid_search5 = grid_search5.fit(X_train, y_train)
best_accuracy_Lregression = grid_search5.best_score_
best_parameters_Lregression = grid_search5.best_params_




'''  XGBOOST '''

from xgboost import XGBClassifier
classifier6 = XGBClassifier(learning_rate=0.12,n_estimators=100)
classifier6.fit(X_train, y_train)

# Predicting the Test set results
y_pred6 = classifier6.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred6)

accuracy_xgboost= (cm6[0][0]+cm6[1][1])/154

from sklearn.model_selection import GridSearchCV
parameters6 = [{'learning_rate': [0.12,0.13,0.125], 'n_estimators': [80,100,125]}]
grid_search6 = GridSearchCV(estimator = classifier6,
                           param_grid = parameters6,
                           scoring = 'accuracy',
                           cv = 10
                           )
grid_search6 = grid_search6.fit(X_train, y_train)
best_accuracyXGBOOST = grid_search6.best_score_
best_parametersxgboost = grid_search6.best_params_



''' USING ANN ALSO KNOWING THE EXACT PROBABILITY OF A PERSON HAVING DIABITIES '''


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier7 = Sequential()

# Adding the input layer and the first hidden layer
classifier7.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier7.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier7.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier7.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier7.fit(X_train, y_train, batch_size = 20 , nb_epoch = 300)

prob_diabities = classifier7.predict(X_test)
y_pred7 = prob_diabities
for jj in range(0,154):
    if(y_pred7[jj]>0.5):
      y_pred7[jj]=1
    else:
        y_pred7[jj]=0
        

from sklearn.metrics import confusion_matrix
cm7=confusion_matrix(y_test,y_pred7)
accuracy_ANN = (cm7[0][0]+cm7[1][1])/154


''' k-fold cross validation '''
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
