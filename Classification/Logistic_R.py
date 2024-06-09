from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataset = datasets.load_digits()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#instancier le modèle
logreg = linear_model.LogisticRegression()
#entraîner le modèle
logreg.fit(X_train, y_train)
#tester le modèle
y_pred = logreg.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
#Afficher la précision du modèle
print("Accuracy of Logistic Regression model is:", metrics.accuracy_score(y_test, y_pred)*100)