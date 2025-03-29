import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model = load_model('model.h5')

## load the encoder and scaler 
with open('labelencoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    ohe_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## App

st.title("Customer Churn Prediction Using ANN")
st.subheader("Predict if a customer will leave the bank or not")
st.write("This app uses a neural network model to predict customer churn based on various features.")

# User Input
geography = st.selectbox("Select Geography",ohe_encoder_geography.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider("Age",18,92)
balance =st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox("Has Credit Card ? ",[0,1])
is_active_member=st.selectbox("Is Active Member ? ",[0,1])

# Prepare Input Data
input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}])

# One-hot encode the categorical features(Geography)
geo_encoded = ohe_encoder_geography.transform([[input_data['Geography'].iloc[0]]]).toarray()
#geo_encoded = ohe_encoder_geography.transform([[input_data['Geography']]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=ohe_encoder_geography.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)


input_data = input_data.drop(columns=['Geography'])

## Encode Gender
input_data['Gender']=label_encoder_gender.transform(input_data['Gender'])

input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

if prediction[0][0] > 0.5 :
    st.write('The customer is likely to churn.')
else :
    st.write('The customer is not likely to churn.')