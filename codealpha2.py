import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download("punkt_tab")

# FAQ questions
questions = [
    "What is Python?",
    "What is a list in Python?",
    "What is a tuple in Python?",
    "What is a dictionary in Python?",
    "What is inheritance in Python?"
]

# FAQ answers
answers = [
    "Python is a high-level, interpreted programming language.",
    "A list is an ordered and mutable collection in Python.",
    "A tuple is an ordered and immutable collection in Python.",
    "A dictionary stores data in key-value pairs.",
    "Inheritance allows one class to acquire properties and methods from another class."
]


# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)

    return " ".join(tokens)


# Preprocess FAQ questions
processed_questions = []

for question in questions:
    processed_questions.append(preprocess(question))


# Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)


# Chatbot function
def chatbot(user_question):

    processed_user_question = preprocess(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()

    best_score = similarity[0][best_match]

    if best_score < 0.2:
        return "Sorry, I don't know the answer to that question."

    return answers[best_match]


# Chat loop
print("🤖 FAQ Chatbot")
print("Type 'exit' to stop the chatbot.")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot(user_input)

    print("Bot:", response)
