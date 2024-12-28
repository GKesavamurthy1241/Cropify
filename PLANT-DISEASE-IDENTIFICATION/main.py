import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tempfile

url = "https://raw.githubusercontent.com/GKesavamurthy1241/Cropify/master/PLANT-DISEASE-IDENTIFICATION/Diseases.png"

try:
    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError if the request failed

    # Open the image as a file-like object
    img = Image.open(BytesIO(response.content))

    # Display image using Streamlit
    st.image(img, caption="Crop Image", use_column_width=True)
def model_prediction(test_image):
    # Raw URL for the model file on GitHub
    model_url = "https://raw.githubusercontent.com/GKesavamurthy1241/Cropify/master/PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras"

    # Fetch the model file
    response = requests.get(model_url)
    if response.status_code == 200:
        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=".keras") as tmp_model_file:
            tmp_model_file.write(response.content)
            model_path = tmp_model_file.name
    else:
        return None

    # Load the model using tf.keras.models.load_model()
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    # Preprocess the image
    try:
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize image to match model input
        input_arr = np.array(image) / 255.0  # Normalize the image
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

    # Make predictions using the loaded model
    try:
        predictions = model.predict(input_arr)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    # Return the index of the highest predicted class
    return np.argmax(predictions)


# Sidebar
st.sidebar.title("Cropify")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    # Add additional content for the Home page if needed

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
    
        # Predict button
        if st.button("Predict"):
            st.snow()  # Show snow animation while processing
            st.write("Our Prediction...")
            result_index = model_prediction(test_image)
            
            if result_index is not None:
                # Define class names
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy']
                
                # Show the prediction result
                st.success(f"Model predicts the image as: {class_name[result_index]}")
