# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 06:20:49 2022

@author: oliviya

"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

hd=pd.read_csv("framingham.csv")

st.write('''
    # Heart Disease Prediction
''')

bg_img = '''<style>
body {
background-image: url("https://th.bing.com/th/id/OIP.gmbpyzsrlII0KFaFa6--KgHaDl?pid=ImgDet&rs=1");
background-size: cover;
}
</style>
'''

st.markdown(bg_img, unsafe_allow_html=True)

st.image("https://th.bing.com/th/id/OIP.gmbpyzsrlII0KFaFa6--KgHaDl?pid=ImgDet&rs=1")

ml_model = pickle.load(open("fmodel.pkl", "rb"))

st.sidebar.header("Input your Data here")
age=st.sidebar.slider("Age", min_value=1, max_value=120, step=1)
currentSmoker=st.sidebar.selectbox("CurrentSmoker",["Yes","No"])
cigsPerDay=st.sidebar.slider("Smoked cigarettes per day",  min_value=0, max_value=70, step=1)
BPMeds=st.sidebar.selectbox("Has Patient been on Blood Pressure Medication?",["Yes","No"])
prevalentStroke=st.sidebar.selectbox("PrevalentStroke",["Yes","No"])
prevalentHyp=st.sidebar.selectbox("Hypertensive?",["Yes","No"])
diabetes= st.sidebar.selectbox("Have diabetes?",["Yes","No"])
totChol=st.sidebar.slider("Cholesterin level",  min_value=30, max_value=300, step=1)
sysBP=st.sidebar.slider("Systolic blood pressure",  min_value=80, max_value=300, step=1)
diaBP=st.sidebar.slider("Diastolic blood pressure",  min_value=30, max_value=150, step=1)
BMI=st.sidebar.slider("BMI",min_value=10, max_value=30, step=1)
heartRate=st.sidebar.slider("HeartRate",min_value=40, max_value=120, step=1)
glucose=st.sidebar.slider("Glucose level", min_value=0, max_value=400, step=1)

pred=pd.DataFrame({"age":age,"currentSmoker":currentSmoker,"cigsPerDay":cigsPerDay,
                   "BPMeds":BPMeds,"prevalentStroke":prevalentStroke, 
                   "prevalentHyp":prevalentHyp, "diabetes":diabetes, 
                   "totChol":totChol,"sysBP":sysBP,"diaBP":diaBP,"BMI":BMI,
                   "heartRate":heartRate,"glucose":glucose},index=[0])

pred["currentSmoker"] = pred['currentSmoker'].apply(lambda x: 1 if x == "Yes" else 0)

pred["prevalentHyp"] = pred['prevalentHyp'].apply(lambda x: 1 if x == "Yes" else 0)

pred['prevalentStroke'] = pred['prevalentStroke'].apply(lambda x: 1 if x == 'Yes' else 0)

pred['diabetes'] = pred['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)

pred['BPMeds'] = pred['BPMeds'].apply(lambda x: 1 if x == 'Yes' else 0)

result=ml_model.predict(pred)
result_percentage=ml_model.predict_proba(pred)
if result[0]==1:
    st.write(round(max(result_percentage[0])*100,2),"% **the patient will develop heart disease**")
else:
    st.write(round(max(result_percentage[0])*100,2),"% **the patient will not develop heart disease**")
    
   
    
    
