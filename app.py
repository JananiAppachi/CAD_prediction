import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Set up the Streamlit UI
st.set_page_config(page_title="Coronary Disease Prediction", layout="wide")

# Sidebar for navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4320/4320334.png", width=120)
st.sidebar.title("🩺 Coronary Disease Prediction")
st.sidebar.markdown("💖 **A simple tool to assess your heart health.**")
st.sidebar.markdown("🔍 Enter your details to check your risk level.")

# Main Page Layout
st.title("💙 Coronary Disease Prediction Web App")
st.write("🚀 Enter your health details and get an AI-powered risk assessment.")



print("Expected Features:", model.feature_names_in_)
print("Expected Shape:", len(model.feature_names_in_))





# User Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol Level (chol)", 100, 400, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (FBS)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 220, 140)
    exang = st.radio("Exercise-Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.slider("Major Vessels Colored (ca)", 0, 4, 0)
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert user input into model format
sex = 1 if sex == "Male" else 0  # Convert gender to numerical
#input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
# One-hot encode categorical variables
sex_encoded = [1, 0] if sex == 0 else [0, 1]  # Male = [0,1], Female = [1,0]
cp_encoded = [1 if cp == i else 0 for i in range(4)]  # Ensure 4 values
fbs_encoded = [1 if fbs == i else 0 for i in range(2)]  # Ensure 2 values
restecg_encoded = [1 if restecg == i else 0 for i in range(3)]  # Ensure 3 values
exang_encoded = [1 if exang == i else 0 for i in range(2)]  # Ensure 2 values

slope_encoded = [1 if slope == i else 0 for i in range(3)]  # Ensure 3 values

ca_encoded = [1 if ca == i else 0 for i in range(5)]  # Ensure 5 values

thal_encoded = [1 if thal == i else 0 for i in range(4)]  # Ensure 4 values


# Combine all features correctly
input_data = np.array([[age, trestbps, chol, thalach, oldpeak, *sex_encoded, *cp_encoded, *fbs_encoded, *restecg_encoded, *exang_encoded, *slope_encoded, *ca_encoded, *thal_encoded]])
input_df = pd.DataFrame(input_data, columns=model.feature_names_in_)
print("Final Input Data Shape:", input_df.shape)  # Should be (1, 30)
st.write("Input Data for Model:", input_data)


# Prediction Button
if st.button("🔍 Predict Risk Level"):
   # Ensure input data has the correct feature order
    input_data = pd.DataFrame(input_data, columns=model.feature_names_in_)  



# Make the prediction
    prediction = model.predict(input_data)

    st.subheader("🩺 Prediction Result:")
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk:** You might have coronary artery disease. Please consult a doctor.")
        st.image("data:image/jpeg;base64,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", caption="Heart Health", use_container_width=True)

    else:
        st.success("✅ **Low Risk:** Your heart health looks good! Keep maintaining a healthy lifestyle.")
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBuUfhqPhMHjqx3Y0KZOhJSeBvJ4PY-zUakQ&s", width=500)


# Preventive Tips Section
st.markdown("## ❤️ How to Prevent Coronary Artery Disease?")
st.markdown("""
- 🏃 **Stay Active:** Exercise for at least 30 minutes daily.  
- 🥗 **Eat Healthy:** Reduce salt, sugar, and unhealthy fats.  
- 🚭 **Avoid Smoking & Alcohol:** Quit smoking and limit alcohol intake.  
- 🌿 **Manage Stress:** Practice meditation, yoga, and deep breathing.  
- 💊 **Regular Checkups:** Visit your doctor for routine heart screenings.  
""")

st.sidebar.markdown("📞 **Emergency Contact:** Call **108** for immediate medical help.")
st.sidebar.markdown("🌍 [Learn More About CAD](https://www.heart.org/en/health-topics/consumer-topics/coronary-artery-disease)")
