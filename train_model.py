# import libraries
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data # features
y = diabetes.target # target variable

# Train model on the entire dataset (for simplicity)
model = LinearRegression() # create an instance of the linear regression model
model.fit(X, y) # fit the model to the entire dataset

joblib.dump(model, 'diabetes_model.pkl') # save the trained model to a file
print("Model printed")