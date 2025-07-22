import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Load model and scaler
model = joblib.load("saved_models/clv_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

st.set_page_config(page_title="CLV Prediction App", layout="wide")
st.markdown("""
    <style>
.success-box {
    background-color: #d1e7dd;
    color: #0f5132;
    padding: 15px;
    border-left: 6px solid #0f5132;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    margin-top: 10px;
    margin-bottom: 10px;
}

    .stApp {
        background-color: #f5f7fa;
        color: #2e2e2e;
        font-family: 'Segoe UI', sans-serif;
    }
    .css-1d391kg, .css-18e3th9 {
        background-color: #e9edf5 !important;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #004d99;
    }
    h1, h2, h3 {
        color: #004080;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Custom alert / result boxes */


    </style>
""", unsafe_allow_html=True)

#header
st.markdown("""
<div style="background-color:#004080;padding:15px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Customer Lifetime Value (CLV) Forecast Dashboard</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar
option = st.sidebar.radio("Select Input Mode", ["Manual Input (Single Customer)", "Upload CSV/Excel (Bulk Prediction)"])

# Helper to preprocess raw dataset
def preprocess_raw_data(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={"Customer ID": "CustomerID"}, inplace=True)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    customer_df = df.groupby('CustomerID').agg({
        'Invoice': 'nunique',
        'Quantity': 'sum',
        'Price': 'mean',
        'InvoiceDate': [
            lambda x: (snapshot_date - x.max()).days,
            lambda x: (x.max() - x.min()).days
        ]
    }).reset_index()

    customer_df.columns = [
        'CustomerID', 'invoice_count', 'total_quantity', 'avg_price', 'recency_days', 'lifespan_days'
    ]
    return customer_df

# Manual input
if option == "Manual Input (Single Customer)":
    st.subheader("Enter Customer Data")
    invoice_count = st.number_input("Number of Invoices", min_value=1, step=1)
    total_quantity = st.number_input("Total Quantity Purchased", min_value=1, step=1)
    avg_price = st.number_input("Average Price per Unit", min_value=0.01, step=0.1)
    recency_days = st.number_input("Days Since Last Purchase", min_value=0, step=1)
    lifespan_days = st.number_input("Customer Lifespan (in days)", min_value=1, step=1)

    if st.button("Predict CLV"):
        input_data = pd.DataFrame([[invoice_count, total_quantity, avg_price, recency_days, lifespan_days]],
                                  columns=['invoice_count', 'total_quantity', 'avg_price', 'recency_days', 'lifespan_days'])
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        st.markdown(f"""
<div style="background-color:#d1e7dd;padding:15px;border-left:6px solid #0f5132;border-radius:8px;margin-top:10px;margin-bottom:10px;color:#0f5132;font-weight:600;">
📈 Predicted 1-Year CLV: £{prediction:.2f}
</div>
""", unsafe_allow_html=True)


# Bulk upload
else:
    st.subheader("Upload Raw Transaction File")
    uploaded_file = st.file_uploader("Upload CSV or Excel file with columns like Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country", type=["csv", "xlsx"])

    if uploaded_file is not None:
        with st.spinner("Processing file, please wait..."):
            if uploaded_file.name.endswith("csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            features_df = preprocess_raw_data(raw_df)
            X = features_df[['invoice_count', 'total_quantity', 'avg_price', 'recency_days', 'lifespan_days']]
            X_scaled = scaler.transform(X)
            features_df['predicted_clv'] = model.predict(X_scaled)

        st.markdown('<div class="success-box">✅ Prediction Complete. Displaying Results</div>', unsafe_allow_html=True)

        st.dataframe(features_df[['CustomerID', 'predicted_clv']])

        st.markdown("## 📈 Strategic CLV Insights")
        
        features_df = pd.merge(features_df, raw_df[['CustomerID', 'Country']].drop_duplicates(), on='CustomerID', how='left')
        features_df = features_df.dropna(subset=['Country'])

        # Segment CLV 
        features_df['clv_segment'] = pd.qcut(features_df['predicted_clv'], q=4, labels=["Low", "Medium", "High", "Very High"])

        # Chart 1: Top 10 Countries by Average CLV
        st.markdown("### 🌍 Top 10 Countries by Average Predicted CLV")
        top_countries = features_df.groupby('Country')['predicted_clv'].mean().sort_values(ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis", ax=ax1)
        ax1.set_xlabel("Average Predicted CLV (£)")
        ax1.set_ylabel("Country")
        st.pyplot(fig1)

        # Chart 2: Quantity vs CLV by Segment (Scatter)
        st.markdown("### 🔁 Quantity vs Predicted CLV (Segmented)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=features_df,
            x='total_quantity',
            y='predicted_clv',
            hue='clv_segment',
            palette='Set2',
            ax=ax2
        )
        ax2.set_xlabel("Total Quantity Purchased")
        ax2.set_ylabel("Predicted CLV (1-Year)")
        ax2.set_title("Customer Behavior: Quantity vs CLV by Segment")
        st.pyplot(fig2)


        csv = features_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", data=csv, file_name="clv_predictions.csv", mime='text/csv')

st.markdown("---")
st.markdown("🔍 **Note**: This app predicts the expected customer lifetime value (CLV) for the next 1 year using purchase patterns and historical transaction data.")
