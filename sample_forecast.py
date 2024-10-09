import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  #type:ignore
import streamlit as st

# Fetching stock data
def fetch_stock_data(ticker):
    data = yf.download(ticker, period='5y')
    return data

# Preprocessing the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit app setup
st.title("Stock Prediction using LSTM")
st.write("""
This app predicts stock prices using an LSTM model trained on historical stock data.
""")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch the stock data
if ticker:
    stock_data = fetch_stock_data(ticker)
    
    st.write(f"### {ticker} Stock Price Data")
    st.line_chart(stock_data['Close'])

    # Preprocess the data
    X, y, scaler = preprocess_data(stock_data)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    real_prices = scaler.inverse_transform(y_test)

    # Plot predictions
    st.write(f"### {ticker} Stock Price Prediction")
    fig, ax = plt.subplots()
    ax.plot(real_prices, label='Real Prices')
    ax.plot(predictions, label='Predicted Prices')
    ax.legend()
    st.pyplot(fig)