import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


df = pd.read_csv('bus_passengers_estimated_fare_time.csv')

label_encoder = LabelEncoder()
df['Bus Stop Start'] = label_encoder.fit_transform(df['Bus Stop Start'])
df['Bus Stop End'] = label_encoder.transform(df['Bus Stop End'])

X = df[['Bus Stop Start', 'Bus Stop End']]
y_fare = df['Estimated Fare']
y_time = df['Estimated Time']

X_train, X_test, y_fare_train, y_fare_test, y_time_train, y_time_test = train_test_split(
    X, y_fare, y_time, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

fare_model = LinearRegression()
time_model = LinearRegression()

fare_model.fit(X_train_scaled, y_fare_train)
time_model.fit(X_train_scaled, y_time_train)

fare_mse = mean_squared_error(y_fare_test, fare_model.predict(X_test_scaled))
time_mse = mean_squared_error(y_time_test, time_model.predict(X_test_scaled))

print(f'Fare Model Mean Squared Error: {fare_mse}')
print(f'Time Model Mean Squared Error: {time_mse}')

joblib.dump(fare_model, 'fare_model.pkl')
joblib.dump(time_model, 'time_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')




import streamlit as st
import pandas as pd
import joblib

fare_model = joblib.load('fare_model.pkl')
time_model = joblib.load('time_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Bus Fare and Time Estimator')


start_stop = st.selectbox('Select Start Bus Stop', options=label_encoder.classes_)
end_stop = st.selectbox('Select End Bus Stop', options=label_encoder.classes_)

if st.button('Estimate'):
    start_encoded = label_encoder.transform([start_stop])[0]
    end_encoded = label_encoder.transform([end_stop])[0]
    input_data = pd.DataFrame([[start_encoded, end_encoded]], columns=['Bus Stop Start', 'Bus Stop End'])
    input_scaled = scaler.transform(input_data)


    fare_estimate = fare_model.predict(input_scaled)[0]
    time_estimate = time_model.predict(input_scaled)[0]

   
    st.write(f'Estimated Fare from {start_stop} to {end_stop}: {fare_estimate:.2f}')
    st.write(f'Estimated Time from {start_stop} to {end_stop}: {time_estimate:.2f} minutes')

