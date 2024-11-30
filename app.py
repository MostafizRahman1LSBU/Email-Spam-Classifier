import streamlit as st 
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')


def Transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    
    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:] #Copying values of variable y to variable message
    y.clear() #remove all items


    for i in message:
        y.append(ps.stem(i))

    return ' '.join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email ", "Via SMS", "Other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = Transform_message(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')
