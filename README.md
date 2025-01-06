<div align="center">
    <h1>CROPIFY: An Intelligent System for Crop Recommendation and Disease Identification!</h1>
</div>

## Features

- **Smart Crop Recommendation :** Utilizes machine learning algorithms to recommend suitable crops based on factors like soil nutrients, climate conditions, and historical data.
- **Plant Disease Identification :** Employs convolutional neural networks (CNNs) to detect and classify plant diseases accurately from uploaded images, enabling timely interventions.
- **User-Friendly Interface :** Offers an intuitive interface for farmers to input relevant data and receive personalized crop recommendations and disease diagnosis.
- **Today's Weather Forecast :** Provides real-time weather insights on temperature, humidity, and more, enabling farmers to plan their farming activities with precision.
- **Smart Farming Guide :** Guides farmers in planning crop planting schedules for optimal productivity based on soil quality, weather, and other essential factors.


# 📌 Crop Recommendation Model

The **Crop Recommendation Model** utilizes machine learning algorithms to suggest the most suitable crops for farmers based on environmental and soil factors. By analyzing data such as soil nutrients, temperature, humidity, pH, and rainfall, the model provides tailored crop recommendations to ensure optimal growth and productivity. The model uses seven classification algorithms, with **Random Forest** achieving the highest accuracy of 99.55%. This helps farmers make informed decisions on crop selection, ensuring better yields and efficient farming practices.

## Dataset

This dataset consists of **2200 rows** in total.
**Each row has 8 columns representing Nitrogen, Phosphorous, Potassium, Temperature, Humidity, PH, Rainfall and Label**
NPK(Nitrogen, Phosphorous and Potassium) values represent the NPK values in the soil. Temperature, humidity and rainfall are the average values of the sorroundings environment respectively. PH is the PH value present in the soil. **The Label column tells us the type of crop that's best suited to grow based on these conditions.
Label is the value we will be predicting**


## Model Architecture
 
For the Crop Recommendation Model, seven classification algorithms were utilized to predict suitable crop recommendations. These algorithms include:

- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest (achieved the best accuracy)
- XGBoost
- KNN
  
Each algorithm was trained on a dataset comprising various factors such as soil nutrients, climate conditions, and historical data to provide accurate crop recommendations to farmers.

## Integration

These two models are integrated into the Smart Crop Recommendation System with Plant Disease Identification. This system provides farmers with comprehensive support, offering both crop recommendations based on various factors and precise identification of crop diseases through image analysis. By combining these models, the system enables farmers to make informed decisions, optimize crop selection, and effectively manage plant diseases for sustainable agriculture and enhanced productivity.

## Results

- Seven classification algorithms were evaluated for crop recommendation tasks.
- The accuracy of each algorithm was assessed, with the Random Forest algorithm achieving the highest accuracy of 99.54%.
- Table 1 below illustrates the accuracy achieved by each algorithm:

**Table 1: Accuracy vs Algorithms**

| Algorithm            | Accuracy   |
|----------------------|------------|
| Decision Tree        | 90.0       |
| Gaussian Naive Bayes| 99.09      |
| Support Vector Machine (SVM) | 10.68 |
| Logistic Regression  | 95.23      |
| Random Forest        | 99.55      |
| XGBoost              | 99.09      |
| KNN                  | 97.5       |

# 📌Plant Disease Identification Model 

The **Plant Disease Identification Model** utilizes Convolutional Neural Networks (CNN) to accurately identify plant diseases from leaf images. Trained on the **Plant Disease Image Dataset**, which includes 70,295 images in the training set and 17,572 images in the validation set, the model covers 38 different plant disease classes across 14 crops. It detects and classifies diseases such as **Apple Scab**, **Tomato Blight**, and **Powdery Mildew**, offering farmers a reliable tool for early disease detection. 

## Dataset

The **Plant Disease Image Dataset**, used for crop disease identification, consists of 70,295 plant images from the training set and 17,572 images from the validation set, covering a variety of 38 different plant disease classes. The images are standardized to a resolution of 128x128 pixels, and the dataset occupies approximately five gigabytes of storage space.


## Model Architecture
   
For the Plant Disease Identification Model, a Convolutional Neural Network (CNN) architecture was employed. This CNN model was specifically trained for crop disease identification. Leveraging deep learning techniques, the CNN analyzes images of plant leaves to detect and classify diseases accurately. This model aids farmers in early disease detection and management, contributing to improved crop health and yield.


### Key Features:
- **Crop Specific**: The model is designed to diagnose diseases for a specific set of crops.
- **Disease Diagnosis**: It can classify diseases based on images of leaves.
- **Accuracy**: The CNN model demonstrates high accuracy in identifying plant diseases, helping farmers and researchers detect issues early.

### Supported Crops and Diseases:
- The model works with a predefined list of 15 crops.
- For each crop, the model is trained to detect and classify up to 38 specific diseases.

Since model is trained for specific crops only so it can diagnose those specific crops only. The List of Crops For which this model will be helpful is:

```
[ 'Apple',
'Blueberry',
'Cherry_(including sour)',
'Corn_(maize)',
'Grape',
'Orange',
'Peach', 'Pepper, _bell',
'Potato',
'Raspberry',
'Soybean',
'Squash',
'Strawberry',
'Tomato' ]
```


The crop which can be used for diagnosis can only diagnose specific disease for which the model is trained. The List of crop diseases on Which Model is trained on is:

```
Found 17572 files belonging to 38 classes.
['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

```
### How it Works:
- The model uses images of plant leaves to detect symptoms of various diseases.
- It applies CNN-based image classification to identify the correct disease for a given crop.
