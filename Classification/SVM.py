import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

#charger les données
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#extraire le min et le max pour optimiser l'affichage du graphe initial (ne pas commencer à partir du 0)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
#apprentissage
Svc_classifier = svm.SVC(kernel='linear', C=1.0).fit(X, y)
Z = Svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)

#personnaliser le graphe d'affichage
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
plt.show()