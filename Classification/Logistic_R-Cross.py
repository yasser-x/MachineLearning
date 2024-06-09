from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the breast cancer dataset
data = load_breast_cancer()
features = data['data']
labels = data['target']

# Split the dataset into features and target variable
X = features
y = labels

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=5000)

# Divide the dataset into 20 parts for cross-validation (1 for training and 19 for testing in each fold)
kfold = KFold(n_splits=20)

# Results of cross-validation
results = cross_val_score(model, X, y, cv=kfold)
print("Cross-validation results:", results)
print("Mean accuracy:", results.mean())
print("Standard deviation:", results.std())
