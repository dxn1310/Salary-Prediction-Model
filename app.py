import streamlit as st
import pickle
import numpy as np

regressor =  pickle.load(open('model.pkl','rb'))
co = pickle.load(open('le_country.pkl','rb'))
ed = pickle.load(open('le_education.pkl','rb'))

st.markdown("<h1 style='text-align: center; color: white;'>Software Developer Salary Prediction</h1>", unsafe_allow_html=True)

st.write("""### We need some information to predict the salary""")

countries = (
    "United States",
    "India",
    "United Kingdom",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
)

education = (
    "Less than a Bachelors",
    "Bachelor’s degree",
    "Master’s degree",
    "Post grad",
)


country = st.selectbox("Country", countries)
education = st.selectbox("Education Level", education)

expericence = st.slider("Years of Experience", 0, 50, 10)

ok = st.button("Calculate Salary")
if ok:
    X = np.array([[country, education, expericence ]])
    X[:, 0] = co.transform(X[:,0])
    X[:, 1] = ed.transform(X[:,1])
    X = X.astype(float)

    salary = regressor.predict(X)
    st.subheader(f"The estimated salary is ${salary[0]:.2f}")