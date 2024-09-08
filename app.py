import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()
# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text, language='english', preserve_line=True)  # Tokenize text

    y = []
    for i in text:
        if i.isalnum():  # Remove non-alphanumeric characters
            y.append(i)

    text = y[:]  # Copy y to text
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Remove stopwords and punctuation
            y.append(i)

    text = y[:]  # Copy y to text
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)

# Load the vectorizer and model

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

# Input message from user
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
