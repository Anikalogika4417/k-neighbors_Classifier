import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


credit_data=pd.read_csv('credit_data.csv')

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

X = np.array(features).reshape(-1, 3)
Y = np.array(target)

X = preprocessing.MinMaxScaler().fit_transform(X)

feature_train, feature_test, target_train, target_test = train_test_split(X, Y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=20)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))