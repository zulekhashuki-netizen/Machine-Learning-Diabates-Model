# import libraries
import numpy as np
import joblib


joblib.dump(model, 'diabetes_model.pkl') # save the trained model to a file
print("Model printed")

#load  the model from the file
model = joblib.load('diabetes_model.pkl')
print("Model loaded")