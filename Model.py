import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('Dataset\heart.csv')

# X is the matrix of independent features.
# y is the vector of Target class.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Spliting the data 80:20 Train:Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train the model (LogisticRegression)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred_logReg = LogReg.predict(X_test)

print("Accuracy LogisticRegression: ", accuracy_score(y_test, y_pred_logReg)*100)

#Train the model (GaussianNB)
from sklearn.naive_bayes import GaussianNB

gaus =  GaussianNB()
gaus.fit(X_train, y_train)

y_pred_gaus = gaus.predict(X_test)

print("Accuracy GaussianNB: ", accuracy_score(y_test, y_pred_gaus)*100)

#Train the model (KNeighborsClassifier)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Accuracy KNeighborsClassifier: ", accuracy_score(y_test, y_pred_knn)*100)

#Train the model (Decision Trees)
from sklearn import tree

dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Accuracy Decision Trees: ", accuracy_score(y_test, y_pred_dt)*100)


#Train the model (RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, criterion = "entropy", random_state=2) #Accuracy RandomForestClassifier:  86.88524590163934
"""param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}"""
#rf = RandomForestClassifier(random_state=2)
#rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)
#rf_random.fit(X_train, y_train)

#print("Best Parameters found: ", rf_random.best_params_)

"""best_rf = rf_random.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best_rf = best_rf.predict(X_test)"""

y_pred_rf = rf.predict(X_test)

print("Accuracy RandomForestClassifier: ", accuracy_score(y_test, y_pred_rf)*100)
#print("Accuracy of Tuned RandomForestClassifier: ", accuracy_score(y_test, y_pred_best_rf)*100) #Accuracy of Tuned RandomForestClassifier:  85.24590163934425

#Train the model (SVC)
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 2)
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
print("Accuracy SVC: ", accuracy_score(y_test, y_pred_svc)*100)