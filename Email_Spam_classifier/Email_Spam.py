import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# NLTK setup 

nltk.download('stopwords')


# Load dataset

# df = pd.read_csv("C:\Users\hp\Desktop\email-spam\Kaggle-SMS-Spam-Collection-Dataset--master\spam.csv", encoding="latin-1")
df = pd.read_csv(
    "C:/Users/hp/Desktop/email-spam/Email_Spam_classifier/spam.csv",
    encoding="latin-1"
)

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


# Label encoding
# ham -> 0, spam -> 1

df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Text preprocessing

ps = PorterStemmer()
corpus = []

for i in range(len(df)):
    msg = df['message'][i]

    # Replace patterns
    msg = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', msg)
    msg = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
    msg = re.sub(r'£|\$', 'moneysymb', msg)
    msg = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', msg)
    msg = re.sub(r'\d+(\.\d+)?', 'numbr', msg)

    # Remove punctuation
    msg = re.sub(r'[^\w\d\s]', ' ', msg)

    # Lowercase
    msg = msg.lower()

    # Tokenization + stemming + stopword removal
    msg = msg.split()
    msg = [ps.stem(word) for word in msg if word not in stopwords.words('english')]

    # Join words back
    msg = ' '.join(msg)

    corpus.append(msg)


# Vectorization

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = df['label'].values


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)


# Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))


# Decision Tree

dt = DecisionTreeClassifier(random_state=50)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
