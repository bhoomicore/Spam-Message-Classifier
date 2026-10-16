import joblib
from pathlib import Path

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load the trained model and TF-IDF vectorizer
model = joblib.load(MODEL_DIR / "spam_classifier.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


def predict_message(message):
    # Convert the message into TF-IDF features
    message_vector = vectorizer.transform([message])

    # Predict the class
    prediction = model.predict(message_vector)[0]

    # Display the result
    if prediction == 1:
        return "SPAM"
    else:
        return "NOT SPAM"


# Interactive testing
print("\n===== SMS SPAM CLASSIFIER =====")
print("Enter 'exit' to stop testing.\n")

while True:
    message = input("Enter a message: ").strip()

    if message.lower() == "exit":
        print("Thank you for testing!")
        break

    if not message:
        print("Please enter a valid message.\n")
        continue

    result = predict_message(message)

    print(f"Prediction: {result}\n")