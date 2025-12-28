import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing

st.title(' 🏰 House Price prediction using ML')
st.image('https://img.freepik.com/free-vector/charming-house-with-tree-illustration_1308-176337.jpg?semt=ais_hybrid&w=740&q=80')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏡 Select House features')
st.sidebar.image('https://img.freepik.com/free-vector/charming-house-with-tree-illustration_1308-176337.jpg?semt=ais_hybrid&w=740&q=80')
all_value = []
for i in X:  
   min_value =  int(X[i].min())
   max_value =  int(X[i].max())
   ans = st.sidebar.slider(f'Select {i} value',min_value, max_value)
   all_value.append(ans)

 # st.write(all_value)

