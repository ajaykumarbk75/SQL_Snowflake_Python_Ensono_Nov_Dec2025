# Mini Project - Ticket Feedback Analysis
import pandas as pd

# Sample ticket dataset
data = {
    'ticket_id': [101, 102, 103, 104, 105, 106, 107],
    'feedback': [
        "Great support and quick resolution",
        "Issue solved, very helpful",
        None,
        "Problem not fixed, very bad experience",
        "",
        "Average service",
        "Excellent service"
    ]
}

df = pd.DataFrame(data)

# 1. Find missing feedback
missing_feedback = df['feedback'].isnull() | (df['feedback'].str.strip() == "")
missing_count = missing_feedback.sum()

# 2. Function to classify feedback
def classify_feedback(text):
    if pd.isnull(text) or text.strip() == "":
        return "Missing"
    text = text.lower()
    if "good" in text or "great" in text or "excellent" in text or "helpful" in text:
        return "Good"
    elif "bad" in text or "not" in text or "poor" in text:
        return "Bad"
    else:
        return "Neutral"

# Apply classification
df['feedback_type'] = df['feedback'].apply(classify_feedback)

# 3. Count feedback types
feedback_counts = df['feedback_type'].value_counts()

# Display results
print("Ticket Feedback Summary\n")
print(df)

print("\nFeedback Counts:")
print(feedback_counts)

print(f"\nNumber of tickets with missing feedback: {missing_count}")
