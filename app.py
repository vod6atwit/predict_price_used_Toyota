import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# load the Random Forest Regression model
final_model = load('./regressor_random_forest.joblib') 

# load the X features for training ColumnTransformer/OneHotEncoder
X_deploy = pd.read_csv('./Models/X_deploy.csv')
X_deploy = np.array(X_deploy)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 2, 4])], remainder='passthrough')
ct.fit_transform(X_deploy)

# Encoding the categorical variables
def encoding(input):
  return ct.transform(input).toarray()

# predict function
def predict(model, input_df):  
    predictions = model.predict(input_df)
    return predictions

  
st.title('Used Toyota Price Prediction App')
model = st.selectbox('Model', sorted([' GT86', ' Corolla', ' RAV4', ' Yaris', ' Auris', ' Aygo', ' C-HR',
       ' Prius', ' Avensis', ' Verso', ' Hilux', ' PROACE VERSO',
       ' Land Cruiser', ' Supra', ' Camry', ' Verso-S', ' IQ',
       ' Urban Cruiser']))
year = st.selectbox('Year', sorted([2016, 2017, 2015, 2020, 2013, 2019, 2018, 2014, 2012, 2005, 2003,
       2004, 2001, 2008, 2007, 2010, 2011, 2006, 2009, 2002, 1999, 2000,
       1998]))
transmission = st.selectbox('Transmission', ['Manual', 'Automatic', 'Semi-Auto', 'Other'])
mileage = st.number_input('Mileage', min_value=2, max_value=174419, value=25000)
fuelType = st.selectbox('FuelType', ['Petrol', 'Other', 'Hybrid', 'Diesel'])
tax = st.number_input('Tax', min_value=0, max_value=565, value=95)
mpg = st.number_input('MPG', min_value=2.8, max_value=235., value=63.)
engineSize = st.selectbox('Engine Size', sorted([2. , 1.8, 1.2, 1.6, 1.4, 2.5, 2.2, 1.5, 1. , 1.3, 0. , 2.4, 3. ,
       2.8, 4.2, 4.5]))

output=""

input_dict = {'model' : model, 'year' : year, 'transmission' : transmission, 'mileage' : mileage, 
              'fuelType' : fuelType, 'tax' : tax, 'mpg' : mpg, 'engineSize' : engineSize}

input_df = pd.DataFrame([input_dict])
input_df = np.array(input_df)

X_input = encoding(input_df)

if st.button("Predict"):
    output = predict(model=final_model, input_df=X_input)[0]
    output = 0 if output<0  else output 
    output = '$' + str(round(output,2))

st.success('The price for this car is {}'.format(output))