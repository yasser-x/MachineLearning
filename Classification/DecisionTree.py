import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.datasets import load_breast_cancer

# Charger les données de cancer du sein
data = load_breast_cancer()
features = data['data']
labels = data['target']
feature_names = data['feature_names']

# Diviser les données en caractéristiques (X) et étiquettes (y)
X = features
y = labels

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Lancer l'apprentissage
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Tester
y_pred = clf.predict(X_test)

# Afficher les résultats
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)

# Visualiser l'arbre de décision
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=feature_names,
                class_names=data['target_names'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('breast_cancer_tree.png')
Image(graph.create_png())
