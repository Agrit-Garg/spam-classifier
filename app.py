import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

st.set_page_config(page_title="Spam Detector", page_icon=None, layout="centered")
def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.header(":blue[Welcome to the Email/SMS Spam Classifier web application!]")
st.write("Enter a message in the text area, click the 'Predict' button, and the web application will tell you if the message is spam or not.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    # 2. vectorize
    # 3. predict
    # 4. Display

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.error("🚨 This message is Spam!")
    else:
        st.success("✅ This message is Not Spam!")

st.markdown(
"""
    \n
    ---
    \n
"""

)

st.markdown("# Model Information")

st.write("I have designed the model that predict whether an email or SMS is spam or not. It employs advanced machine learning techniques to analyze the content of messages and make predictions.")

# Key features
st.markdown("## Key Features")
st.write("- **Accuracy:** The model achieves an accuracy of 97% in identifying spam messages.")
st.write("- **Precision:** The precision of the model is 1, indicating high accuracy in identifying true positives.")
st.write("- **Algorithm:** The model uses the Multinomial Naive Bayes algorithm, a popular choice for text classification tasks.")

# Limitations
st.markdown("## Limitations")
st.write("It's important to note that while my model performs exceptionally well, it may not be 100% accurate. Machine learning models are based on patterns learned from data, and there can be cases where the model provides incorrect predictions. Users should be aware of this limitation.")

# How the model works
st.markdown("## How It Works")
st.write("The model utilizes natural language processing (NLP) techniques and the Multinomial Naive Bayes algorithm. It analyzes the text of messages, extracts relevant features, and makes predictions based on its training.")


# Evaluation metrics
st.markdown("## Evaluation Metrics")
st.write("During training, the model was evaluated using various metrics such as accuracy, precision, recall, and F1 score to ensure its effectiveness in different scenarios.")

# Closing statement
st.write("If you have any questions or concerns about my model, feel free to reach out to me. I value your feedback.")

st.markdown(
"""
    ---
    Developed by :blue[Agrit Garg]. \n
    Data source: :blue[Kaggle]
"""

)





