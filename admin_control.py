import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@st.cache_data  
def load_data():
    df = pd.read_csv('bus_passengers_delhi.csv')
   
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute
   
    label_encoder_stop = LabelEncoder()
    df['Bus Stop'] = label_encoder_stop.fit_transform(df['Bus Stop'])

    label_encoder_action = LabelEncoder()
    df['Action'] = label_encoder_action.fit_transform(df['Action'])
    return df, label_encoder_stop, label_encoder_action
df, label_encoder_stop, label_encoder_action = load_data()

X = df[['Time', 'Bus Stop', 'Action']]
y = df['Number of Passengers']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title('Bus Scheduling and Route Management System')
st.write('This application predicts the number of passengers boarding or deboarding at a specific bus stop based on time and action.')

time_input = st.number_input('Time (in minutes since midnight)', min_value=0, max_value=1439, value=720)
bus_stop_input = st.selectbox('Bus Stop', options=label_encoder_stop.classes_)
action_input = st.selectbox('Action', options=['Boarding', 'Deboarding'])

bus_stop_encoded = label_encoder_stop.transform([bus_stop_input])[0]
action_encoded = label_encoder_action.transform([action_input])[0]
input_features = np.array([[time_input, bus_stop_encoded, action_encoded]])
input_features_scaled = scaler.transform(input_features)

predicted_passengers = model.predict(input_features_scaled)
st.write(f'Predicted number of passengers: {predicted_passengers[0]}')
