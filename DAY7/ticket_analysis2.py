import streamlit as st
import pandas as pd

# -----------------------------
# App Title
# -----------------------------
st.title("🎫 Case Ticket Feedback Analysis")
st.write("Mini Project – Feedback Classification Dashboard")

# -----------------------------
# Sample Ticket Dataset
# -----------------------------
data = {
    'ticket_id': [101, 102, 103, 104, 105, 106, 107],
    'feedback': [
        "Great Support and quick resolution",
        "Issue resolved, very helpfull",
        None,
        "Problem not fixed, very bad expereince",
        "",
        "Average service",
        "Exellent Service"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Show Raw Data
# -----------------------------
st.subheader("📄 Ticket Feedback Data")
st.dataframe(df)

# -----------------------------
# Find Missing Feedback
# -----------------------------
missing_feedback = df['feedback'].isnull() | (df['feedback'].str.strip() == "")
missing_count = missing_feedback.sum()

# -----------------------------
# Feedback Classification Function
# -----------------------------
def classify_feedback(text):
    if pd.isnull(text) or text.strip() == "":
        return "Missing"
    text = text.lower()
    if "good" in text or "great" in text or "excellent" in text or "helpfull" in text:
        return "Good"
    elif "bad" in text or "not" in text or "poor" in text:
        return "Bad"
    else:
        return "Neutral"

# -----------------------------
# Apply Classification
# -----------------------------
df['feedback_type'] = df['feedback'].apply(classify_feedback)

# -----------------------------
# Display Classified Data
# -----------------------------
st.subheader("✅ Classified Feedback")
st.dataframe(df)

# -----------------------------
# Feedback Summary
# -----------------------------
feedback_counts = df['feedback_type'].value_counts()

st.subheader("📊 Feedback Summary")
st.bar_chart(feedback_counts)

# -----------------------------
# Metrics
# -----------------------------
st.subheader("📌 Key Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Tickets", len(df))

with col2:
    st.metric("Missing Feedback", missing_count)

# -----------------------------
# Footer
# -----------------------------
st.info("📘 This is a basic NLP-style feedback analysis using keyword rules.")
