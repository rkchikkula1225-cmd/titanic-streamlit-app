import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter a passenger's details to estimate the chance of survival. "
         "The model is a logistic regression trained on the Titanic training data.")

st.sidebar.header("Passenger details")
pclass = st.sidebar.selectbox("Ticket class", [1, 2, 3], index=2,
                              format_func=lambda x: {1: "1st (upper)", 2: "2nd (middle)", 3: "3rd (lower)"}[x])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age (years)", 0.0, 80.0, 30.0, step=1.0)
sibsp = st.sidebar.number_input("Siblings / spouses aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents / children aboard", 0, 6, 0)
fare = st.sidebar.number_input("Fare paid (£)", 0.0, 520.0, 32.0, step=1.0)
embarked = st.sidebar.selectbox("Port of embarkation", ["S", "C", "Q"],
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

passenger = pd.DataFrame([{
    "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
    "Parch": parch, "Fare": fare, "Embarked": embarked,
}])

st.subheader("Passenger entered")
st.dataframe(passenger, hide_index=True)

if st.button("Predict survival", type="primary"):
    prob = float(model.predict_proba(passenger)[0, 1])
    survived = prob >= 0.5
    if survived:
        st.success(f"Prediction: **Survived** (probability {prob:.1%})")
    else:
        st.error(f"Prediction: **Did not survive** (probability of survival {prob:.1%})")
    st.progress(prob)
    st.caption("Probability of survival according to the model. "
               "This is a statistical estimate from historical data, not a certainty.")
