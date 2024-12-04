import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

st.title("ETF Price Prediction with LSTM")

# Sidebar inputs
st.sidebar.header("Configure Analysis")
etf_ticker = st.sidebar.text_input("ETF Ticker", "NIFTYBEES.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
time_step = st.sidebar.slider("Time Step", min_value=10, max_value=120, value=60, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=20, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)

# Fetch data
st.write(f"Fetching data for {etf_ticker}...")
data = yf.download(etf_ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data found for the given ticker and date range.")
else:
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Data Preprocessing
    st.write("Preprocessing data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    st.write("Building and training LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training model..."):
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

    st.success("Model training complete!")

    # Predict and visualize
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    st.write("Visualizing predictions...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[time_step + 1:], y=data['Close'][time_step + 1:], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=data.index[time_step + 1:], y=predictions.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(
        title="ETF Analysis with Deep Learning",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
