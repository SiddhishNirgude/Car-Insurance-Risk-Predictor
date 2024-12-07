import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer  # Use this instead of importing KNNImputer directly

# Page configuration
st.set_page_config(
    page_title="Car Insurance Risk Predictor - Data Science",
    page_icon="🚗",
    layout="wide"
)

# Cache the data loading
@st.cache_data
def load_data():
    # Add your data loading logic here
    return pd.read_csv("your_data.csv")

def main():
    st.title("Car Insurance Risk Predictor - Data Science Analysis")
    
    # Sidebar navigation for Data Science sections
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Section",
        ["Data Overview", 
         "Data Statistics",
         "Data Merging and Missingness",
         "EDA",
         "Correlation Analysis",
         "Category Analysis and Hypothesis Testing"]
    )

    # Load data
    try:
        df = load_data()
    except:
        st.error("Error loading data. Please check data file path.")
        return

    # Section routing
    if page == "Data Overview":
        data_overview()
    elif page == "Data Statistics":
        data_statistics(df)
    elif page == "Data Merging and Missingness":
        data_merging_missingness(df)
    elif page == "EDA":
        exploratory_data_analysis(df)
    elif page == "Correlation Analysis":
        correlation_analysis(df)
    elif page == "Category Analysis and Hypothesis Testing":
        category_analysis(df)

def data_overview():
    st.header("Data Overview 📊")
    
    st.write("""
    ### Data Sources
    This project combines data from three main sources:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1. Insurance Data
        - Customer demographics
        - Claim history
        - Risk factors
        [Source Link](your_link_here)
        """)
    
    with col2:
        st.markdown("""
        #### 2. Vehicle Features
        - Technical specifications
        - Safety features
        - Performance metrics
        [Source Link](your_link_here)
        """)
    
    with col3:
        st.markdown("""
        #### 3. Maintenance Records
        - Service history
        - Repair records
        - Maintenance scores
        [Source Link](your_link_here)
        """)

def data_statistics(df):
    st.header("Data Statistics 📈")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Data info
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

def data_merging_missingness(df):
    st.header("Data Merging and Missingness Analysis 🔍")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    
    # Plot missing values
    fig = px.bar(
        missing_df[missing_df['Missing Values'] > 0],
        title='Missing Values by Column',
        labels={'value': 'Percentage Missing', 'index': 'Columns'}
    )
    st.plotly_chart(fig)
