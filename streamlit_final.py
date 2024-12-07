import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer

# Page configuration
st.set_page_config(
    page_title="TechImpact Solutions - Car Insurance Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# Function to show introduction content (will be used both in landing and tab)
def show_introduction_content():
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
        st.image("Imageof-Auto-Insurance.jpg", caption="Car Insurance Analytics")

    # Data Sources
    st.header("Data Sources and Integration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1. Insurance Claims Data
        **Source**: [Car Insurance Claim Data](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data)
        
        **Contains**:
        - Customer demographics
        - Claims history
        - Policy details
        - Risk indicators
        """)

    with col2:
        st.markdown("""
        ### 2. Vehicle Specifications
        **Source**: [Car Insurance Claim Prediction](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification?select=train.csv)
        
        **Contains**:
        - Technical specifications
        - Safety features
        - Vehicle characteristics
        - Performance metrics
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

    # Integration Process
    st.header("Data Integration Process")
    st.markdown("""
    Our comprehensive approach includes:
    - Data merging using common identifiers
    - Handling missing values and duplicates
    - Feature engineering from multiple sources
    - Quality checks and validation
    
    This enables analysis of:
    - ✅ Risk factors from multiple perspectives
    - ✅ Relationships between vehicle features and claims
    - ✅ Impact of maintenance on insurance risk
    """)

# Load data (cached)
@st.cache_data
def load_data():
    try:
        final_integrated_df_cleaned = pd.read_csv("final_integrated_dataset.csv")
        return final_integrated_df_cleaned
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the CSV file is in the correct directory.")
        return None

# Load the data
df = load_data()

# Initially show the introduction content
show_introduction_content()

# Space selection in sidebar
st.sidebar.title("TechImpact Solutions")
selected_space = st.sidebar.selectbox(
    "Select Space",
    ["Select Space", "Production Space", "Data Science Space"]
)

# Show content based on space selection
if selected_space == "Data Science Space":
    # Create tabs for Data Science Space
    tabs = st.tabs([
        "Introduction",
        "Data Overview",
        "Data Statistics",
        "Data Merging & Missingness",
        "EDA",
        "Correlation Analysis",
        "Category Analysis"
    ])
    
    # Content for each tab
    with tabs[0]:  # Introduction
        show_introduction_content()
        
    with tabs[1]:  # Data Overview
        st.title("Data Overview")
        st.header("Dataset Description")
        # Add your data overview content here
        if df is not None:
            st.dataframe(df.head())
            st.write(f"Dataset Shape: {df.shape}")
            
    with tabs[2]:  # Data Statistics
        st.title("Data Statistics")
        if df is not None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            st.header("Numeric Features Statistics")
            st.dataframe(df[numeric_cols].describe())
            
    with tabs[3]:  # Data Merging & Missingness
        st.title("Data Merging and Missingness Analysis")
        if df is not None:
            missing = df.isnull().sum()
            st.write("Missing Values Summary:")
            st.write(missing[missing > 0])
            
    # Add content for remaining tabs as needed
    
elif selected_space == "Production Space":
    st.info("Production Space features are under development.")

# Stop if data loading failed
if df is None:
    st.stop()
