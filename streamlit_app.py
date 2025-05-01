import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load models
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return kmeans_model, scaler

# Set page config
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Customer Segmentation Analysis")
st.markdown("""
This app uses machine learning to segment customers based on their purchasing behavior.
Upload a CSV file with customer data to get predictions.
""")

# Load models
try:
    kmeans_model, scaler = load_models()
    cluster_labels = {0: 'Lookers/Surfers', 1: 'Impulse Customers', 2: 'Loyal Customers'}
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Store customer info
        customer_info = df[['Customer ID', 'Customer Name']] if 'Customer Name' in df.columns else df[['Customer ID']]
        
        # Define the expected features (same as in training)
        expected_features = ['Quantity', 'Sales', 'Discount', 'Profit', 'returned_numeric', 'customer_frequency', 'purchase_speed']
        
        # Check if all expected features are present
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required features: {', '.join(missing_features)}")
            st.stop()
        
        # Select only the expected features
        df_features = df[expected_features]
        
        # Handle missing values
        df_cleaned = df_features.dropna()
        
        # Scale the data
        scaled_data = scaler.transform(df_cleaned)
        
        # Make predictions directly on scaled data
        predictions = kmeans_model.predict(scaled_data)
        
        # Create results dataframe
        results_df = pd.DataFrame()
        if 'Customer Name' in customer_info.columns:
            results_df['Customer Name'] = customer_info['Customer Name']
        else:
            results_df['Customer ID'] = customer_info['Customer ID']
        
        results_df['Predicted Segment'] = [cluster_labels[i] for i in predictions]
        
        # Display results
        st.subheader("Customer Segmentation Results")
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the table
            st.dataframe(results_df, use_container_width=True)
        
        with col2:
            # Create pie chart
            segment_counts = results_df['Predicted Segment'].value_counts()
            # Convert NumPy int64 to Python int for the pie chart
            segment_values = [int(x) for x in segment_counts.values]
            fig = px.pie(
                values=segment_values,
                names=segment_counts.index,
                title="Customer Segment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button for results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

# Add sample data download
st.sidebar.header("Sample Data")
st.sidebar.markdown("""
Download a sample CSV file to test the application.
The file contains the required columns for customer segmentation.
""")
sample_data = pd.read_csv('nt_customer_segmentation.csv')
csv = sample_data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Sample CSV",
    data=csv,
    file_name="sample_customer_data.csv",
    mime="text/csv"
) 