from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to disk
joblib.dump(model, 'iris_model.pkl')
