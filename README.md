# 🌾 Agro Forecast: Time Series-Based Crop Price and Demand Prediction

## 🧭 Introduction
Agriculture is one of the most important sectors in India. Predicting crop prices and demand can help farmers, traders, and policymakers make better market decisions.  
**Agro Forecast** is a machine learning project that uses **Time Series Analysis (LSTM)** to forecast crop prices and demand trends across multiple Indian markets.

The system provides insights about future prices based on past trends, helping in efficient market planning and reducing financial risks for farmers.

---

## 🎯 Objective
- Develop a **Deep Learning-based system** to predict crop prices using **LSTM (Long Short-Term Memory)**.
- Analyze market-wise and commodity-wise **price trends and correlations**.
- Assist **farmers, traders, and policymakers** in making data-driven decisions.
- Deploy a **Streamlit web app** for real-time forecasting and visualization.

---

## 📊 Dataset Information
The dataset simulates real agricultural market data from **6 markets** and **6 commodities** across **5 years (2020–2024)**.

**File:** `agro_dataset_multi_market.csv`  
**Rows:** ~65,000  
**Columns:**
| Column | Description |
|---------|-------------|
| `date` | Date of record |
| `commodity` | Crop name (e.g., Tomato, Wheat, Rice, etc.) |
| `market` | Local market name (e.g., Pune, Nashik, Mumbai, etc.) |
| `modal_price` | Average market price (₹ per Quintal) |
| `arrivals` | Quantity arrivals in market (Tons) |

> You can extend this dataset to include weather features like rainfall, temperature, and humidity for better forecasting accuracy.

---

## ⚙️ Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| **Programming Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Web App Framework** | Streamlit |
| **Deployment** | Streamlit Cloud / Hugging Face Spaces |

---

## 🧠 Model Workflow
1. **Data Loading:** Import CSV dataset using Pandas  
2. **Preprocessing:** Convert dates, filter commodity & market, and scale prices  
3. **Sequence Creation:** Prepare 30-day input windows for LSTM  
4. **Model Building:** Two-layer LSTM with dropout for regularization  
5. **Training:** 25 epochs using Adam optimizer (loss = MSE)  
6. **Prediction:** Forecast next 7 days’ prices  
7. **Visualization:** Actual vs Predicted and Future Forecast plots  

---

## 📈 Sample Analysis & Insights
- Market and commodity-wise **price trends visualization**
- **Correlation analysis** between arrivals and price  
- **Future price forecast** for selected market and crop  
- **RMSE & R² evaluation** to check model accuracy  

---

