import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Charger le dataset de cancer du sein
data = datasets.load_breast_cancer()
X = data.data[:, np.newaxis, 2]  # Utilisation d'une seule caractéristique pour la régression
y = data.target

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer le modèle de régression linéaire
regr = linear_model.LinearRegression()

# Entraîner le modèle en utilisant les données d'entraînement
regr.fit(X_train, y_train)

# Faire des prédictions en utilisant les données de test
y_pred = regr.predict(X_test)

# Afficher les coefficients
print('Coefficients: \n', regr.coef_)

# Afficher l'erreur quadratique moyenne
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Afficher le score de variance
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Tracer les données de test et les prédictions
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression on Breast Cancer Data')
plt.show()
