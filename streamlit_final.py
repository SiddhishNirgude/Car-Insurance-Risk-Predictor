import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer  # Use this instead of importing KNNImputer directly


# Set page configuration
st.set_page_config(page_title="Car Insurance Risk Predictor", layout="wide")

# Custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #FAF3E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data loading function
@st.cache_data
def load_data():
    try:
        # Load your datasets
        final_integrated_df_cleaned = pd.read_csv("final_integrated_dataset.csv")
        return final_integrated_df_cleaned
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the CSV file is in the correct directory.")
        return None

# Load the data
df = load_data()

# Check if data was loaded successfully
if df is None:
    st.stop()
else:
    st.success("Data loaded successfully!")

# Display the car insurance illustration
st.image("Imageof-Auto-Insurance.jpg", 
         caption="Car Insurance Risk Prediction", use_column_width=True)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Introduction", 
         "Data Overview",
         "Data Statistics",
         "Data Merging & Missingness",
         "EDA",
         "Correlation Analysis",
         "Category Analysis"]

selected_page = st.sidebar.selectbox("Go to", pages)

# Introduction page content
if selected_page == "Introduction":
    # Main title with emoji
    st.title("🚗 Car Insurance Risk Predictor")
    
    # Project overview section
    st.header("Project Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        This project focuses on developing a sophisticated risk prediction system 
        for car insurance claims by analyzing three key aspects:
        
        1. **Insurance Claims Analysis**: Understanding patterns in insurance claims
        2. **Vehicle Safety Assessment**: Evaluating safety features and their impact
        3. **Maintenance Pattern Study**: Analyzing how maintenance affects risk
        """)
        
    with col2:
        # Add your project image
        st.image("Imageof-Auto-Insurance.jpg", caption="Car Insurance Analytics")
    
    # Project components
    st.header("Key Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Insurance Risk Analysis
        - Claims prediction
        - Risk factor identification
        - Premium calculation insights
        """)
        
    with col2:
        st.markdown("""
        ### 🛡️ Vehicle Safety Analysis
        - Safety feature evaluation
        - Accident risk assessment
        - Safety rating impact
        """)
        
    with col3:
        st.markdown("""
    ### 3. Maintenance Records
    **Source**: [Vehicle Maintenance Data](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)
    
    **Contains**:
    - Service history
    - Maintenance patterns
    - Repair records
    - Vehicle condition metrics
    """)

    # Add data integration process
    st.subheader("Data Integration Process")
    st.markdown("""
    The integration process involved:
    - Merging datasets using common identifiers
    - Handling missing values and duplicates
    - Feature engineering from multiple sources
    - Quality checks and validation

    This comprehensive approach allows us to analyze:
    - ✅ Risk factors from multiple perspectives
    - ✅ Relationships between vehicle features and claims
    - ✅ Impact of maintenance on insurance risk
    """)

    # Add acknowledgment
    st.markdown("""
    ---
    *Data Sources: All datasets are sourced from Kaggle, a platform for data science and machine learning enthusiasts. 
    We acknowledge the original contributors of these datasets.*
    """)

    # Project objectives
    st.header("Project Objectives")
    st.markdown("""
    - Develop accurate risk prediction models
    - Identify key factors influencing insurance claims
    - Provide actionable insights for risk assessment
    - Create an interactive tool for risk evaluation
    """)
    
    # Data sources
    st.header("Data Integration")
    st.write("""
    This project combines data from three main sources to provide comprehensive insights:
    1. Insurance claim records
    2. Vehicle specifications and safety features
    3. Maintenance and service history
    """)
    
    # Project objectives
    st.header("Project Objectives")
    st.markdown("""
    - Develop accurate risk prediction models
    - Identify key factors influencing insurance claims
    - Provide actionable insights for risk assessment
    - Create an interactive tool for risk evaluation
    """)

# 1. Data Overview Page
if page == 'Data Overview':
    st.title('Data Overview')
    
    st.header("Dataset Description")
    st.write("This project integrates three main data sources:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚗 Insurance Data
        - Customer demographics
        - Claims history
        - Risk factors
        - Policy information
        """)
        
    with col2:
        st.markdown("""
        ### 🔧 Vehicle Features
        - Technical specifications
        - Safety features
        - Performance metrics
        - Vehicle characteristics
        """)
        
    with col3:
        st.markdown("""
        ### 📋 Maintenance Data
        - Service history
        - Maintenance scores
        - Vehicle condition
        - Repair records
        """)

    # Display sample data
    st.header("Sample Data")
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape}")

# 2. Data Statistics Page
elif page == 'Data Statistics':
    st.title('Data Statistics')
    
    # Numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    # Summary statistics for numeric columns
    st.header("Numeric Features Statistics")
    st.dataframe(df[numeric_cols].describe())
    
    # Categorical columns summary
    st.header("Categorical Features Summary")
    for col in categorical_cols:
        st.subheader(f"{col} Distribution")
        st.write(df[col].value_counts())
        st.write(df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# 3. Data Merging and Missingness
elif page == 'Data Merging and Missingness':
    st.title('Data Merging and Missingness Analysis')
    
    # Missingness heatmap
    st.header("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    st.pyplot(fig)
    
    # Missing values summary
    st.header("Missing Values Summary")
    missing = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage': df.isnull().sum() / len(df) * 100
    })
    missing = missing[missing['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
    st.dataframe(missing)
