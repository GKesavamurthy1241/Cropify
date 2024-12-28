import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from PIL import Image
from io import BytesIO

# URL of the image
url = "https://raw.githubusercontent.com/GKesavamurthy1241/Cropify/master/CROP-RECOMMENDATION/crop.png"

try:
    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError if the request failed

    # Open the image as a file-like object
    img = Image.open(BytesIO(response.content))

    # Display image using Streamlit
    st.image(img, caption="Crop Image", use_column_width=True)
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching the image: {e}")
except Exception as e:
    st.error(f"Error opening the image: {e}")

# URL for the CSV data
url = "https://raw.githubusercontent.com/GKesavamurthy1241/Cropify/master/CROP-RECOMMENDATION/Crop_recommendation.csv"

try:
    # Read the CSV file from the URL
    df = pd.read_csv(url)
except Exception as e:
    st.error(f"Error loading the CSV file: {e}")

# Features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForest model
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)

# Save the model using Pickle
RF_pkl_filename = 'RF.pkl'
with open(RF_pkl_filename, 'wb') as RF_Model_pkl:
    pickle.dump(RF, RF_Model_pkl)

# Load the trained model
RF_Model_pkl = pickle.load(open('RF.pkl', 'rb'))

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # Making predictions using the trained model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

# Streamlit code for the web app interface
def main():  
    # Setting the title of the web app
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("Cropify")
    app_mode = st.sidebar.selectbox("Select Page", ["HOME", "CROP RECOMMENDATION"])

    # Adding content to "CROP RECOMMENDATION" page
    if app_mode == "CROP RECOMMENDATION":
        st.sidebar.header("Enter Crop Details")
        
        # Sidebar inputs for environmental factors
        nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
        humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
        
        inputs = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]  # Input array for prediction
        
        # Validate inputs and make prediction
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        if st.sidebar.button("Predict"):
            # Check for invalid input (any zero or NaN values)
            if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
                st.error("Please fill in all input fields with valid values before predicting.")
            else:
                prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                st.success(f"The recommended crop is: {prediction[0]}")  # Display recommended crop

# Running the main function
if __name__ == '__main__':
    main()
