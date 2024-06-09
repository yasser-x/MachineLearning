import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# Charger les données de cancer du sein
data = load_breast_cancer()
features = data['data']
labels = data['target']

# Réduction de dimension pour la visualisation (utilisation de PCA pour réduire à 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

# Affichage initial des données réduites
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=20)
plt.title('Visualization of Breast Cancer Data (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Application de l'apprentissage KMeans
kmeans = KMeans(n_clusters=2)  # Il y a deux classes dans le dataset de cancer du sein
kmeans.fit(X_pca)

# Tester la prédiction
y_kmeans = kmeans.predict(X_pca)

# Affichage graphique des centres de clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.title('KMeans Clustering of Breast Cancer Data (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
