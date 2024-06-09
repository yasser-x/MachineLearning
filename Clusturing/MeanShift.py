import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

style.use("ggplot")

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

# Lancer l'apprentissage
ms = MeanShift()
ms.fit(X_pca)

# Affichage des centres des clusters
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print("Cluster centers:\n", cluster_centers)

# Affichage des clusters
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)

colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']

plt.figure(figsize=(8, 6))
for i in range(len(X_pca)):
    plt.plot(X_pca[i][0], X_pca[i][1], colors[labels[i]], markersize=5)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="o", color='k', s=100, linewidths=5, zorder=10)
plt.title('MeanShift Clustering of Breast Cancer Data (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
