import pandas as pd
import re
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project Topic Database

project_topics = [
    {
        "title": "AI Chatbot for Student Support",
        "domain": "NLP",
        "skills": "Python NLP Flask",
        "difficulty": "Beginner"
    },

    {
        "title": "Plant Disease Detection",
        "domain": "Computer Vision",
        "skills": "Python TensorFlow OpenCV",
        "difficulty": "Advanced"
    },
    {
    "title": "Sentiment Analysis of Customer Reviews",
    "domain": "Natural Language Processing",
    "skills": "Python NLTK NLP Machine Learning",
    "difficulty": "Beginner"
},

{
    "title": "Student Performance Prediction",
    "domain": "Machine Learning",
    "skills": "Python Pandas Scikit-learn Regression",
    "difficulty": "Intermediate"
},

{
    "title": "Face Recognition Attendance System",
    "domain": "Computer Vision",
    "skills": "Python OpenCV Deep Learning",
    "difficulty": "Advanced"
},

{
    "title": "Sales Forecasting System",
    "domain": "Data Science",
    "skills": "Python Pandas Time Series Machine Learning",
    "difficulty": "Intermediate"
},

{
    "title": "Email Spam Detection",
    "domain": "NLP",
    "skills": "Python NLP TF-IDF Scikit-learn",
    "difficulty": "Beginner"
},

{
    "title": "Stock Price Prediction",
    "domain": "Finance AI",
    "skills": "Python Machine Learning Time Series",
    "difficulty": "Advanced"
}
]


topics_df = pd.DataFrame(project_topics)

print(topics_df)


# Student Profile

student_profile = {
    "name": "Supraja",
    "interests": "NLP Machine Learning Recommendation System",
    "skills": "Python Pandas Scikit-learn",
    "academic": "B.Tech Computer Science"
}

print(student_profile)
# -----------------------------
# Text Preprocessing Function
# -----------------------------

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub("[^a-zA-Z]", " ", text)

    words = nltk.word_tokenize(text)

    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(cleaned_words)


# Test Text Preprocessing

sample = "I love Machine Learning and Python Projects!"

cleaned = clean_text(sample)

print("Original Text:")
print(sample)

print("\nCleaned Text:")
print(cleaned)
topics_df = pd.DataFrame(project_topics)
# Combine project information into one text

topics_df["project_text"] = (
    topics_df["title"] + " " +
    topics_df["domain"] + " " +
    topics_df["skills"]
)

print(topics_df["project_text"])
topics_df["clean_project_text"] = (
    topics_df["project_text"].apply(clean_text)
)

print(topics_df["clean_project_text"])
# Create TF-IDF model

vectorizer = TfidfVectorizer()

project_vectors = vectorizer.fit_transform(
    topics_df["clean_project_text"]
)

print(project_vectors)# -----------------------------
# Student Text Preparation
# -----------------------------

student_text = (
    student_profile["interests"] + " " +
    student_profile["skills"] + " " +
    student_profile["academic"]
)

clean_student_text = clean_text(student_text)

print("\nStudent Clean Text:")
print(clean_student_text)
# Convert student profile into vector

student_vector = vectorizer.transform(
    [clean_student_text]
)

print(student_vector)
# Calculate similarity score

similarity_scores = cosine_similarity(
    student_vector,
    project_vectors
)

print("\nSimilarity Scores:")
print(similarity_scores)
# Convert similarity matrix into a list

scores = similarity_scores[0]
# Add scores to dataframe

topics_df["match_score"] = scores
# Sort highest matching projects first

recommendations = topics_df.sort_values(
    by="match_score",
    ascending=False
)
print("\n🎯 Recommended Project Topics")
print("="*40)

top_projects = recommendations.head(5)


for index, project in top_projects.iterrows():

    print("\nProject:", project["title"])
    print("Domain:", project["domain"])
    print("Skills:", project["skills"])
    print("Difficulty:", project["difficulty"])
    print(
        "Match Score:",
        round(project["match_score"] * 100, 2),
        "%"
    )
# -----------------------------
# Difficulty Level Filtering
# -----------------------------

level = "Beginner"

filtered_projects = recommendations[
    recommendations["difficulty"] == level
]


print("\n\n🔍 Filtered Projects -", level)
print("="*40)


for index, project in filtered_projects.iterrows():

    print("\nProject:", project["title"])
    print("Domain:", project["domain"])
    print("Skills:", project["skills"])
    print("Match Score:",
          round(project["match_score"] * 100, 2),
          "%")
