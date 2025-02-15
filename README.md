 "Empower Your Heart Health"
 "Predict. Prevent. Prosper. Take control of your heart health today."
 Coronary Artery Disease Prediction through Machine Learning
Overview:
This project aims to enhance the prediction of Coronary Artery Disease (CAD) using machine learning algorithms. CAD is a major cause of mortality worldwide, and early detection is crucial for improving patient outcomes. This project leverages various machine learning models to predict CAD, providing non-invasive solutions and contributing to better healthcare management.

Dataset:
The project utilizes the UCI Heart Disease dataset, which includes the following features:

Age: Age of the patient
Sex: Gender of the patient (0 = male, 1 = female)
CP: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
Trestbps: Resting blood pressure (mm Hg)
Chol: Serum cholesterol (mg/dl)
Fbs: Fasting blood sugar (>120 mg/dl) (1 = true, 0 = false)
Restecg: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
Thalach: Maximum heart rate achieved
Exang: Exercise induced angina (1 = yes, 0 = no)
Oldpeak: ST depression induced by exercise relative to rest
Slope: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
Ca: Number of major vessels colored by fluoroscopy
Thal: Thallium stress test result (0 = normal, 1 = fixed defect, 2 = reversible defect, 3 = not described)
Target: Presence of CAD (0 = no disease, 1 = disease present)
Methodology

Preprocessing:

Data Type Conversion
Renaming Columns
Handling Missing Values
Outlier Detection
One-Hot Encoding
Standard Scaling
Train-Test Split

Model Development:

Decision Tree
Random Forest Algorithm
Support Vector Machine (SVM)
Linear Regression
K-Nearest Neighbors (KNN)

Performance Metrics:

Accuracy Score
Mean Squared Error
Precision
Confusion Matrix

Results
The performance of various algorithms on the dataset is as follows:

Algorithm	Accuracy (%)
Linear Regression	81.82
K-Nearest Neighbors (KNN)	81.82
Support Vector Machine (SVM)	90.26
Decision Tree Classifier	97.08
Random Forest Classifier	98.05
The Random Forest Classifier achieved the highest accuracy, demonstrating its effectiveness in predicting CAD.

Conclusion
This project highlights the utility of machine learning algorithms in predicting CAD with high accuracy. The Random Forest Classifier proved to be the most effective model among those evaluated. Future work includes expanding the dataset and refining preprocessing techniques to further improve prediction accuracy.


## 🌐 Live App Link  
🎯 Check out the deployed app: [Coronary Risk Check](https://coronary-risk-check.streamlit.app/)

