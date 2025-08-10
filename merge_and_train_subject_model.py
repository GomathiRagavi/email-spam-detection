import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Load Spambase dataset
# -----------------------------
# Column names from spambase.names file
columns = [
    # 48 word_freq features
    "word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our",
    "word_freq_over","word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail",
    "word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses",
    "word_freq_free","word_freq_business","word_freq_email","word_freq_you","word_freq_credit",
    "word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp",
    "word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs",
    "word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
    "word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct",
    "word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re",
    "word_freq_edu","word_freq_table","word_freq_conference",
    # 6 char_freq features
    "char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#",
    # Capital run length features
    "capital_run_length_average","capital_run_length_longest","capital_run_length_total",
    # Class label
    "label"
]

file_path = r"M:\rise intern\email spam detection\spambase.csv"
spambase = pd.read_csv(file_path, header=None, names=columns)

# Convert numeric features to pseudo-text
def spambase_to_text(row):
    words = []
    for col in spambase.columns[:-1]:  # all except label
        freq = row[col]
        if freq > 0:
            word = col.replace("word_freq_", "").replace("char_freq_", "char_")
            words.extend([word] * int(freq * 10))  # scale to make counts more distinct
    return " ".join(words)

spambase_texts = spambase.apply(spambase_to_text, axis=1)
spambase_df = pd.DataFrame({
    "subject": spambase_texts,
    "label": spambase["label"]
})

# -----------------------------
# 2. Load Ling-Spam dataset
# -----------------------------


lingspam = pd.read_csv(r"lingspam.csv\lingspam.csv")
 # must exist with subject, label columns
lingspam_df = pd.DataFrame({
    "subject": lingspam["subject"],
    "label": lingspam["label"]
})

# -----------------------------
# 3. Merge datasets
# -----------------------------
combined_df = pd.concat([spambase_df, lingspam_df], ignore_index=True)
print(f"Total samples: {len(combined_df)}")
print(combined_df["label"].value_counts())

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    combined_df["subject"].fillna(""),
    combined_df["label"],
    test_size=0.2,
    random_state=42,
    stratify=combined_df["label"]
)

# -----------------------------
# 5. Vectorize text
# -----------------------------
vectorizer = TfidfVectorizer()
# Replace NaN with empty string
X_train = X_train.fillna("")
X_test = X_test.fillna("")

# Remove rows where subject is still empty after fill
train_mask = X_train.str.strip() != ""
X_train = X_train[train_mask]
y_train = y_train[train_mask]

test_mask = X_test.str.strip() != ""
X_test = X_test[test_mask]
y_test = y_test[test_mask]

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6. Train classifier
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# -----------------------------
# 8. Save model + vectorizer
# -----------------------------
joblib.dump(model, "spam_subject_model.pkl")
joblib.dump(vectorizer, "subject_vectorizer.pkl")
