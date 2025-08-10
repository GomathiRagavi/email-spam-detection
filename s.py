import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("spam_subject_model.pkl")
vectorizer = joblib.load("subject_vectorizer.pkl")

# Test subject
test_subject = ["S, Win PPIs, ₹2 Lakh Prize"]

# Transform using the same vectorizer
X_input = vectorizer.transform(test_subject)

# Prediction
pred = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0]

print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
print(f"Probabilities -> Ham: {proba[0]:.4f}, Spam: {proba[1]:.4f}")

# Show top contributing words
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

# Only keep features that appear in the test input
nonzero_indices = X_input.nonzero()[1]
contribs = sorted(
    [(feature_names[i], coefs[i]) for i in nonzero_indices],
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\nTop contributing words for this prediction:")
for word, weight in contribs[:10]:
    print(f"{word:20} {weight:+.4f}")
