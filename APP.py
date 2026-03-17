# import libraries
import joblib
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# load the diabetes dataaset
diabetes = load_diabetes()
X = diabetes.data # features
y = diabetes.target # target variable

# Train model on the entire dataset (for simplicity)
model = LinearRegression() # create an instance of the linear regression model
model.fit(X, y) # fit the model to the entire dataset

# define title and description
st.write("Zullu the ML Engineer")

st.title("Diabetes Dataset Linear Regression")
st.write("" \
"This is a simple linear regression model for predicting diabetes progression." \
" The dataset contains 10 features and a target variable that represents the progression of diabetes." \
" The Machine learning algorithm used is Linear Regression" 
"The purpose of this model is to predict the progression of diabetes based on the input features."
)

# Enter values for the features
st.write("Please enter values for the following features:")
age = st.number_input("Age")
sex = st.number_input("Sex")
bmi = st.number_input("Body Mass Index")     
bp = st.number_input("Average Blood Pressure")
s1 = st.number_input("S1")  
s2 = st.number_input("S2")
s3 = st.number_input("S3")
s4 = st.number_input("S4")
s5 = st.number_input("S5") 
s6 = st.number_input("S6") 

# prediction
st.write("Predicted Diabetes Progression:")

# define conditional statement to make prediction when all input values are provided
if age and sex  and bmi and bp and s1 and s2 and s3 and s4 and s5 and s6:
    input_data = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]) # create a 2D array for the input data
    model = LinearRegression() # create an instance of the linear regression model
    model.fit(X, y) # fit the model to the entire dataset




# prediction results
st.title("Predict")

# Predict only when button is clicked
if st.button("Predict"):
    prediction = model.predict([input_data])
    st.write("The predicted diabetes progression is:", prediction[0])

# refresh the page to clear input values
if st.button("Refresh"):
    st.rerun()

# output results
st.write("This prediction is based on the input features provided and the linear regression model trained on the diabetes dataset.")




