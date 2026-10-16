
import os
import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

DATA_PATH = "dataset/SMSSpamCollection"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Please check the dataset folder."
    )

data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# Convert labels: spam = 1, ham = 0
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove any missing or invalid rows
data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].astype(int)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print(data["label"].value_counts())


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


data["message"] = data["message"].apply(clean_text)

X = data["message"]
y = data["label"]


# --------------------------------------------------
# 3. Split into training, validation and test sets
# --------------------------------------------------

# First: 80% development data, 20% final test data
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Next: 75% of development for training, 25% for validation
# Overall proportions: 60% training, 20% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("\nTraining messages:", len(X_train))
print("Validation messages:", len(X_val))
print("Test messages:", len(X_test))


# --------------------------------------------------
# 4. TF-IDF vectorization
# --------------------------------------------------

# Learn vocabulary and IDF weights from training data only
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation and test data without fitting again
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 5. Train and compare algorithms
# --------------------------------------------------

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train on training data only
    model.fit(X_train_tfidf, y_train)

    trained_models[name] = model

    # Validation predictions for model selection
    val_predictions = model.predict(X_val_tfidf)

    val_precision = precision_score(
        y_val, val_predictions, zero_division=0
    )

    val_recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    val_f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    print("Validation precision:", round(val_precision, 4))
    print("Validation recall:", round(val_recall, 4))
    print("Validation F1:", round(val_f1, 4))

    results.append({
        "Model": name,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,
        "Validation F1": val_f1
    })


# --------------------------------------------------
# 6. Select the best model
# --------------------------------------------------

# Prioritize spam precision.
# Use validation F1 as a tie-breaker.
results_df = pd.DataFrame(results)

best_model_name = (
    results_df.sort_values(
        by=["Validation Precision", "Validation F1"],
        ascending=False
    ).iloc[0]["Model"]
)

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 7. Refit selected model on training + validation
# --------------------------------------------------

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

# Fit a fresh vectorizer on the combined development data
final_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)

X_train_final_tfidf = final_vectorizer.fit_transform(
    X_train_final
)

X_test_final_tfidf = final_vectorizer.transform(X_test)

final_model = models[best_model_name]

final_model.fit(X_train_final_tfidf, y_train_final)


# --------------------------------------------------
# 8. Evaluate on the held-out test data
# --------------------------------------------------

# Test both algorithms for comparison.
# Model selection was already performed using validation data.
test_results = []

for name, model in trained_models.items():

    # Retrain each algorithm on all development data
    comparison_model = models[name]

    comparison_model.fit(
        X_train_final_tfidf,
        y_train_final
    )

    test_predictions = comparison_model.predict(
        X_test_final_tfidf
    )

    test_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Spam Precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "Spam Recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "Spam F1": f1_score(
            y_test, test_predictions, zero_division=0
        )
    })

    if name == best_model_name:
        final_model = comparison_model
        final_predictions = test_predictions


test_results_df = pd.DataFrame(test_results)

print("\nFINAL TEST RESULTS")
print(test_results_df.round(4).to_string(index=False))

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["Ham", "Spam"],
    zero_division=0
))


# --------------------------------------------------
# 9. Confusion matrix visualization
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=[0, 1]
)

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")
plt.show()


# --------------------------------------------------
# 10. Save the selected model and vectorizer
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    final_model,
    "models/spam_classifier.pkl"
)

joblib.dump(
    final_vectorizer,
    "models/tfidf_vectorizer.pkl"
)

# Save test results for the README
test_results_df.to_csv(
    "outputs/test_metrics.csv",
    index=False
)

print("\nModel and vectorizer saved successfully!")
print("Confusion matrix saved in outputs/")
print("Test metrics saved in outputs/test_metrics.csv")
