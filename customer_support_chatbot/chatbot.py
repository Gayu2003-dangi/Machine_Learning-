import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
patterns = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])

# Convert text to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Chat function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]

    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

# Chat loop
print("Customer Support Chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
