
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Title and introduction
st.title("ETF Analysis and Prediction")
st.write("This app fetches ETF data, preprocesses it, and uses an LSTM model for prediction.")

# Ticker input
ticker = st.text_input("Enter the ETF Ticker (e.g., NIFTYBEES.NS):", "NIFTYBEES.NS")
start_date = st.date_input("Start Date:", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime("2024-11-27"))

if st.button("Fetch Data"):
    # Fetching data
    with st.spinner("Fetching data..."):
        etf_data = yf.download(ticker, start=start_date, end=end_date)
        if etf_data.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            st.success("Data fetched successfully!")
            st.write("Sample Data:")
            st.dataframe(etf_data.head())
            
            # Plotting closing price
            st.write("Closing Price Over Time:")
            fig, ax = plt.subplots()
            etf_data['Close'].plot(ax=ax, title=f"{ticker} Closing Price", xlabel="Date", ylabel="Price")
            st.pyplot(fig)
            
            # Preprocessing
            etf_data = etf_data.dropna()
            scaler = MinMaxScaler(feature_range=(0, 1))
            etf_data['Close_scaled'] = scaler.fit_transform(etf_data[['Close']])

            # Sequence creation
            sequence_length = st.slider("Sequence Length for LSTM:", min_value=30, max_value=365, value=180)
            data = etf_data['Close_scaled'].values
            X, y = [], []
            for i in range(sequence_length, len(data)):
                X.append(data[i-sequence_length:i])
                y.append(data[i])
            X, y = np.array(X), np.array(y)
            
            st.write("Prepared Data:")
            st.write(f"Input Shape: {X.shape}, Output Shape: {y.shape}")
            
            # Splitting data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model training
            if st.button("Train Model"):
                with st.spinner("Training LSTM model..."):
                    model = Sequential([
                        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                        Dropout(0.2),
                        LSTM(units=50, return_sequences=False),
                        Dropout(0.2),
                        Dense(units=1)
                    ])
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)
                    st.success("Model trained successfully!")
                    
                    # Loss plot
                    st.write("Training Loss:")
                    fig, ax = plt.subplots()
                    ax.plot(history.history['loss'], label='Train Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.legend()
                    st.pyplot(fig)

# Footer
st.write("Developed with Streamlit and TensorFlow")
