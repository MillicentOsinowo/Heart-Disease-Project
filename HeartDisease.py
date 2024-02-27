import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('heart (4).csv')
# data.drop('HeartDisease', axis = 1, inplace = True)

df = data.copy()

st.markdown("<h1 style='text-align: center; color: #151965;'>Heart Disease PREDICTOR MODEL</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; margin-top: 0rem; color: #32407B;'>BUILT BY OSINOWO MILLICENT</h4>", unsafe_allow_html=True)

st.image('pngwing.com (5).png', width=250, use_column_width=True)
st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>Project Overview</h4>", unsafe_allow_html=True)
st.markdown("<p>The primary objective is to create a resilient machine learning model capable of accurately predicting the likelihood of heart disease based on historical trends and key influencing factors. The project aims to accomplish this by analyzing historical health data and integrating relevant features, including patient demographics, medical history, and lifestyle choices. The ultimate aim is to provide a tool that supports decision-making in the healthcare sector. </p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


sel = ['Age', 'Cholesterol', 'MaxHR', 'RestingBP', 'ChestPainType', 'Oldpeak']

df = df[sel]

# Instantiate a dictionary to hold the transformers
encoded = {}
scaled = {}

for i in df.columns:
    encode = LabelEncoder()
    if df[i].dtypes == 'O':
        df[i] = encode.fit_transform(df[i])
        encoded[i+'_encoded'] = encode
    else:
        scale = StandardScaler()
        df[i] = scale.fit_transform(df[[i]])
        scaled[i + '_scaled'] = scale

x = df
y = data.HeartDisease

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, stratify=y)

model = LogisticRegression()
model.fit(xtrain, ytrain)


st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>PREDICTOR MODEL</h4>", unsafe_allow_html=True)
# st.dataframe(data)

st.sidebar.image('pngwing.com (6).png', width=100, use_column_width=True, caption='Hello, Lets Have A Heart to Heart Session!')
st.markdown("<br>", unsafe_allow_html=True)

Age = st.sidebar.number_input('AGE', data['Age'].min(), data['Age'].max())
Cholesterol = st.sidebar.number_input('Cholesterol Level', data['Cholesterol'].min(), data['Cholesterol'].max())
MaxHR = st.sidebar.number_input('Heartrate/Pulse', data['MaxHR'].min(), data['MaxHR'].max())
RestingBP = st.sidebar.number_input('Blood Pressure', data['RestingBP'].min(), data['RestingBP'].max())
ChestPainType = st.sidebar.selectbox('Chest Pain Type', data.ChestPainType.unique())
Oldpeak = st.sidebar.number_input('Depression induced by exercise compared to rest', data.Oldpeak.min(), data.Oldpeak.max())

inputs = pd.DataFrame({
    'Age': [Age],
    'Cholesterol': [Cholesterol],
    'MaxHR' : [MaxHR],
    'RestingBP': [RestingBP],
    'ChestPainType': [ChestPainType],
    'Oldpeak': [Oldpeak]
})

st.dataframe(inputs)

inputs['Age'] = scaled['Age_scaled'].transform(inputs[['Age']] )
inputs['Cholesterol'] = scaled['Cholesterol_scaled'].transform(inputs[['Cholesterol']])
inputs['MaxHR'] = scaled['MaxHR_scaled'].transform(inputs[['MaxHR']] )
inputs['RestingBP'] = scaled['RestingBP_scaled'].transform(inputs[['RestingBP']])
inputs['ChestPainType'] = encoded['ChestPainType_encoded'].transform(inputs['ChestPainType'])
inputs['Oldpeak'] = scaled['Oldpeak_scaled'].transform(inputs[['Oldpeak']])



st.markdown("<br>", unsafe_allow_html=True)


prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push to Results')
    if pred:
        # Include the prediction step here
        predicted = model.predict(inputs)
        output = 'Have an Healthy Heart' if predicted[0] == 0 else ' Consult A Physician Heart is not in good State'
        st.success(f'The individual is predicted to {output}')
        st.snow()


import plotly.express as px
fig = px.pie(names=['Jiggy Jiggy...That is an Healthy Heart right there!!!','Please Consult your Physician as Your Heart is not in good State'], values=[80, 20], title='Prediction Distribution')
st.plotly_chart(fig)




