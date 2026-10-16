# Spam Message Classifier using NLP

## Project Overview

This project implements a machine learning-based SMS Spam
Message Classifier using Natural Language Processing (NLP).

The system classifies text messages into two categories:

- Spam: Unwanted or potentially fraudulent messages.
- Ham: Legitimate messages.

Two machine learning algorithms are trained and evaluated:
Multinomial Naive Bayes and Logistic Regression.

The models are compared using accuracy, precision, recall,
F1-score, and confusion matrix analysis.

The project also supports interactive predictions on
custom user-input messages.

## Objectives

- Build a text classification system using NLP.
- Clean and preprocess text messages.
- Convert text into numerical features using TF-IDF.
- Compare Multinomial Naive Bayes and Logistic Regression.
- Evaluate the models using held-out test data.
- Visualize classification results.
- Predict whether a custom message is spam or legitimate.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Git and GitHub

## Dataset

The project uses the SMS Spam Collection dataset.

Dataset source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains text messages labeled as spam or ham.

## Text Preprocessing

The preprocessing and feature extraction pipeline includes:

1. Lowercasing text.
2. Tokenization.
3. Stop-word removal.
4. TF-IDF vectorization.

The TF-IDF vectorizer is fitted using training data only
to avoid information leakage from the held-out test set.

## Machine Learning Algorithms

### 1. Multinomial Naive Bayes

A probabilistic classification algorithm commonly used
for text classification.

### 2. Logistic Regression

A supervised classification algorithm used to predict
whether a message belongs to the spam or ham category.

## Model Evaluation

Both models are evaluated on held-out test data.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | TODO | TODO | TODO | TODO |
| Logistic Regression | TODO | TODO | TODO | TODO |

Replace the TODO values with the actual results
printed by train_model.py.

### Evaluation Metrics

- Accuracy: Percentage of correctly classified messages.
- Precision: Proportion of predicted spam messages that
  are actually spam.
- Recall: Proportion of actual spam messages identified.
- F1-score: Harmonic mean of precision and recall.

High spam precision is important because it reduces the
number of legitimate messages incorrectly marked as spam.

## Confusion Matrix

The project generates a confusion matrix visualization
to examine classification performance.

See the confusion matrix image in the outputs directory.

## Interactive Message Prediction

Run the prediction script:

    python predict.py

Enter a message when prompted.

The classifier returns one of the following predictions:

- SPAM
- NOT SPAM

Type `exit` to stop the program.

## Sample Predictions

Record the actual predictions produced by predict.py.

| Input Message | Predicted Class |
|---|---|
|  Win a free mobile phone. Click here to claim your reward! |NOT SPAM |
|WINNER!! You have been selected to receive a FREE prize reward. Call 09061701461 now to claim your cash prize. Valid for 12 hours only!|  SPAM |
|FREE entry in 2 a weekly competition to win FA Cup final tickets. Text WIN to 80086 now! | Add actual prediction |
| Add your tested message | Add actual prediction |
| Add your tested message | Add actual prediction |

These examples demonstrate inference on custom messages.
They are separate from held-out test-set evaluation.

## Project Structure

    Spam_Message_Classifier/
    ├── dataset/
    ├── models/
    ├── outputs/
    ├── train_model.py
    ├── predict.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

Generated model files and evaluation outputs are stored
in their respective directories.

## Installation

Install the required dependencies:

    pip install -r requirements.txt

Train the models and generate evaluation outputs:

    python train_model.py

Run interactive predictions:

    python predict.py

## Conclusion

This project demonstrates a complete NLP text classification
workflow, from preprocessing and feature extraction to model
comparison, evaluation, and interactive prediction.

The model comparison helps identify an appropriate classifier
while considering spam precision and recall.