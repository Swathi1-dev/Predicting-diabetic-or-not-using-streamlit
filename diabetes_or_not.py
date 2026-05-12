import numpy as np
import pickle

import streamlit as st


# Load the trained model from the file
with open("saved_model.sav", "rb") as file:
    model = pickle.load(file)


def diabetes_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    st.title("Diabetes Prediction Web App")
    st.write(
        "Enter the following details to predict whether a person is diabetic or not:"
    )
    Pregnencies = st.number_input(
        "Number of Pregnancies", min_value=0, max_value=20, value=0
    )
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=0)
    BloodPressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=0
    )
    SkinThickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=0
    )
    Insulin = st.number_input("Insulin level", min_value=0, max_value=900, value=0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0
    )
    Age = st.number_input("Age", min_value=0, max_value=120, value=0)

    # code for predictions
    diagnosis = ""

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [
                Pregnencies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
            ]
        )
    st.success(diagnosis)


if __name__ == "__main__":
    main()
