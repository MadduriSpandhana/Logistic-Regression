#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import streamlit as st 
import joblib

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    Age = st.sidebar.number_input('Insert Age', min_value=1, max_value=100, value=30)
    SibSp = st.sidebar.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
    Parch = st.sidebar.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)
    Fare = st.sidebar.number_input('Insert Fare', min_value=0.0, max_value=600.0, value=30.0)
    Embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

    # One-hot encoding for categorical features
    data = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'female': 1 if Sex == 'Female' else 0,
        'male': 1 if Sex == 'Male' else 0,
        'Embarked_C': 1 if Embarked == 'C' else 0,
        'Embarked_Q': 1 if Embarked == 'Q' else 0,
        'Embarked_S': 1 if Embarked == 'S' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features 

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

# Load pre-trained Logistic Regression model
@st.cache_resource
def load_model():
    return joblib.load('logistic_model.pkl')

clf = load_model()

# Make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Survived' if prediction_proba[0][1] > 0.5 else 'Did Not Survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




