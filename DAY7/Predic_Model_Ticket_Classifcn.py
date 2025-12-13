# Predictive Model - Ticket Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset: ticket descriptions and types
data = {
    'description': [
        "Unable to login to email",
        "Request for new laptop",
        "System crash when saving file",
        "Password reset needed",
        "Printer not working",
        "Need access to shared folder",
        "Application not responding",
        "Request for software installation"
    ],
    'type': [
        "incident",
        "request",
        "incident",
        "request",
        "incident",
        "request",
        "problem",
        "request"
    ]
}

df = pd.DataFrame(data)

# Features and target
X = df['description']
y = df['type']

# Convert text to numerical features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict type of new tickets
new_tickets = [
    "Computer not starting",
    "Request for VPN access",
    "Application freezing frequently"
]
new_vectorized = vectorizer.transform(new_tickets)
predictions = model.predict(new_vectorized)

for ticket, pred in zip(new_tickets, predictions):
    print(f"Ticket: '{ticket}' --> Predicted type: {pred}")
