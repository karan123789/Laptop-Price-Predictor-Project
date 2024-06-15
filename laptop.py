import streamlit as st
import numpy as np
import pandas as pd

# Mock DataFrame to replace df.pkl
data = {
    'Company': ['Dell', 'HP', 'Acer', 'Apple', 'Asus', 'Lenovo', 'MSI'],
    'TypeName': ['Ultrabook', 'Gaming', 'Notebook', 'Netbook', 'Workstation', 'Ultrabook', 'Gaming'],
    'Cpu brand': ['Intel Core i5', 'Intel Core i7', 'AMD Ryzen 5', 'AMD Ryzen 7', 'Intel Core i5', 'Intel Core i7', 'AMD Ryzen 5'],
    'Gpu brand': ['Nvidia', 'AMD', 'Intel', 'Nvidia', 'AMD', 'Intel', 'Nvidia'],
    'os': ['Windows', 'MacOS', 'Linux', 'No OS', 'Windows', 'MacOS', 'Linux']
}
df = pd.DataFrame(data)

# Mock model to replace pipe.pkl
class MockModel:
    def predict(self, query):
        # Encoding categorical variables
        encoding = {val: idx for idx, val in enumerate(np.unique(query))}
        encoded_query = [encoding[val] if val in encoding else val for val in query]
        # Mock prediction logic (replace with your model's logic)
        return [np.log(np.sum(encoded_query) + 1000)]  # Return a list with one element

# Instantiate mock model
pipe = MockModel()

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size', min_value=1.0, step=0.1)

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    try:
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    except ZeroDivisionError:
        st.error("Screen size cannot be zero.")
        ppi = 0

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Mock processing of the query (replace this with your actual model processing)
    query = query.reshape(1, -1)[0]  # Flatten to 1D array for encoding
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
