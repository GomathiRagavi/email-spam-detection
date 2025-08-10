import joblib

# --------------------------
# Step 1: Load Model + Vectorizer
# --------------------------
# These should be saved during training using joblib.dump()
model = joblib.load("spam_subject_model.pkl")
vectorizer = joblib.load("subject_vectorizer.pkl")

# --------------------------
# Step 2: Get Input Email
# --------------------------
print("\n📨 Enter your email message below:")
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)
input_email = "\n".join(lines)

# --------------------------
# Step 3: Transform & Predict
# --------------------------
X_input = vectorizer.transform([input_email])  # Convert text to TF-IDF features
prediction = model.predict(X_input)[0]

print("\n🔍 Prediction:", "🚫 SPAM" if prediction == 1 else "✅ HAM (Not Spam)")
