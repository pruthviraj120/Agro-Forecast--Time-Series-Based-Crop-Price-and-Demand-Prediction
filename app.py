import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Agro Forecast Dashboard", layout="wide")
st.title("🌾 Agro Forecast: Advanced Crop Price Prediction Dashboard")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    st.success("✅ Dataset Uploaded Successfully!")

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("Filters")
    commodity = st.sidebar.selectbox("Select Commodity", df['commodity'].unique())
    market = st.sidebar.selectbox("Select Market", df['market'].unique())
    
    # Optional: Select date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    
    # Filter data
    data = df[(df['commodity']==commodity) & (df['market']==market)]
    data = data[(data['date'] >= pd.to_datetime(date_range[0])) & 
                (data['date'] <= pd.to_datetime(date_range[1]))].sort_values('date')

    st.subheader(f"📈 Price Trend: {commodity} in {market}")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data['date'], data['modal_price'], color='green', label='Actual Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Statistical Insights
    # -------------------------------
    st.subheader("📊 Statistical Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Price (₹)", round(data['modal_price'].mean(), 2))
    col2.metric("Max Price (₹)", round(data['modal_price'].max(), 2))
    col3.metric("Min Price (₹)", round(data['modal_price'].min(), 2))

    if 'arrivals' in data.columns:
        corr = data[['modal_price','arrivals']].corr().iloc[0,1]
        st.write(f"💡 Correlation between Price & Arrivals: **{corr:.2f}**")
    
    # -------------------------------
    # Scale and Prepare LSTM Data
    # -------------------------------
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['modal_price']])

    def create_sequences(data, n_steps=30):
        X, y = [], []
        for i in range(len(data)-n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = 30
    X, y = create_sequences(data_scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -------------------------------
    # LSTM Model
    # -------------------------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    with st.spinner("⏳ Training LSTM Model..."):
        model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    st.success("✅ Model Trained Successfully!")

    # -------------------------------
    # Forecast Future Prices
    # -------------------------------
    def predict_future(model, data_scaled, scaler, n_steps=30, days_ahead=7):
        last_seq = data_scaled[-n_steps:]
        preds = []
        for _ in range(days_ahead):
            X_input = last_seq.reshape((1,n_steps,1))
            pred_scaled = model.predict(X_input)
            preds.append(pred_scaled[0][0])
            last_seq = np.append(last_seq[1:], pred_scaled)
        return scaler.inverse_transform(np.array(preds).reshape(-1,1))

    days_ahead = st.slider("🔮 Forecast Days", 3, 15, 7)
    future_preds = predict_future(model, data_scaled, scaler, days_ahead=days_ahead)
    last_date = data['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=days_ahead+1, freq='D')[1:]

    # -------------------------------
    # Forecast Plot
    # -------------------------------
    st.subheader("📊 Forecast: Next Days Prices")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(data['date'][-30:], data['modal_price'].iloc[-30:], label='Actual', color='blue')
    ax2.plot(future_dates, future_preds, label='Forecast', color='red', linestyle='--', marker='o')
    ax2.set_xlabel("Date"); ax2.set_ylabel("Price (₹)")
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)

    # -------------------------------
    # Forecast Table & Download
    # -------------------------------
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (₹)': future_preds.flatten()})
    st.write("📅 Forecast Table")
    st.dataframe(forecast_df.style.format({'Predicted Price (₹)':'{:.2f}'}))

    # Download button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download Forecast CSV", data=csv, file_name=f'{commodity}_{market}_forecast.csv', mime='text/csv')

else:
    st.info("👆 Please upload a CSV file to begin.")
