# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data - small toy dataset (emails and labels)
emails = [
    "Congratulations, you won a free ticket!",
    "Hey, are we still meeting tomorrow?",
    "Claim your free prize now",
    "Important update about your account",
    "You have won a lottery, click to claim",
    "Can we reschedule our meeting?",
    "Free coupons available, hurry up!",
    "Let's catch up over coffee",
]

# Labels: 1 = spam, 0 = not spam
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Test with new emails
new_emails = [
    "Win a brand new car now",
    "Can you send me the report?",
]

new_X = vectorizer.transform(new_emails)
new_pred = model.predict(new_X)

for email, pred in zip(new_emails, new_pred):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"Email: '{email}' -> Prediction: {label}")
