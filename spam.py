import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load spambase dataset
import pandas as pd

# Using raw string to avoid backslash issues
data = pd.read_csv("spambase.csv", header=None)





X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]   # Last column is label: 1 = spam, 0 = ham

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline (scaling + model)
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, "spam_model.pkl")
print("✅ Model saved as spam_model.pkl")

