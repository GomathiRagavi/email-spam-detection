import joblib
import numpy as np

# Load trained model & vectorizer
model = joblib.load("spam_subject_model.pkl")
vectorizer = joblib.load("subject_vectorizer.pkl")

def explain_prediction(text):
    # Vectorize input
    X_input = vectorizer.transform([text])
    
    # Predict probabilities
    probs = model.predict_proba(X_input)[0]
    ham_prob, spam_prob = probs
    label = "Spam" if spam_prob >= 0.5 else "Ham"
    
    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Probabilities -> Ham: {ham_prob:.4f}, Spam: {spam_prob:.4f}")
    
    # Show top contributing words
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    indices = X_input.nonzero()[1]
    word_weights = [(feature_names[i], coefs[i]) for i in indices]
    word_weights_sorted = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop contributing words:")
    for word, weight in word_weights_sorted:
        print(f"{word:20} {weight:+.4f}")

if __name__ == "__main__":
    print("Spam/Ham Classifier — type 'exit' to quit")
    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() == "exit":
            break
        explain_prediction(user_input)
