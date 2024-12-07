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

# Utility function to load data (cached)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_integrated_dataset.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the CSV file is in the correct directory.")
        return None

# Load the data
df = load_data()

# --- TOP-LEVEL NAVIGATION ---
st.sidebar.title("TechImpact Solutions")
selected_space = st.sidebar.selectbox(
    "Select Top-Level Space",
    ["Home", "Data Science Space", "Production Space"]
)

# --- SPACE-SPECIFIC NAVIGATION ---
selected_page = None
if selected_space == "Data Science Space":
    selected_page = st.sidebar.radio(
        "Navigate Data Science Space",
        ["Data Overview", "Data Statistics", "Data Merging & Missingness", 
         "EDA", "Correlation Analysis", "Category Analysis"]
    )
elif selected_space == "Production Space":
    selected_page = st.sidebar.radio(
        "Navigate Production Space",
        ["Risk Assessment", "Vehicle Comparison", "Maintenance Predictor", "Insurance Calculator"]
    )

# --- HOME PAGE ---
def show_home_page():
    st.title("🚗 Car Insurance Risk Predictor")
    st.header("Welcome to TechImpact Solutions")
    st.write("""
    Explore comprehensive tools for analyzing and predicting car insurance risks. 
    Select a space from the dropdown menu to begin:
    - **Data Science Space**: Dive into data analysis and visualization.
    - **Production Space**: Access tools for risk assessment and predictions.
    """)

# --- DATA SCIENCE SPACE PAGES ---
def show_data_overview():
    st.title("Data Overview")
    if df is not None:
        st.write("### Dataset Snapshot:")
        st.dataframe(df.head())
        st.write(f"**Dataset Shape**: {df.shape}")
    else:
        st.error("Data not loaded. Please check the source file.")

def show_data_statistics():
    st.title("Data Statistics")
    if df is not None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        st.write("### Descriptive Statistics for Numeric Features:")
        st.dataframe(df[numeric_cols].describe())
    else:
        st.error("Data not loaded. Please check the source file.")

def show_missingness_analysis():
    st.title("Data Merging & Missingness")
    if df is not None:
        st.write("### Missing Values Summary:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.write(missing)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            st.pyplot(fig)
        else:
            st.success("No missing values found in the dataset!")
    else:
        st.error("Data not loaded. Please check the source file.")

# Placeholder functions for additional pages
def show_eda():
    st.title("Exploratory Data Analysis")
    st.write("EDA content coming soon!")

def show_correlation_analysis():
    st.title("Correlation Analysis")
    st.write("Correlation analysis content coming soon!")

def show_category_analysis():
    st.title("Category Analysis")
    st.write("Category analysis content coming soon!")

# --- PRODUCTION SPACE PAGES ---
def show_risk_assessment():
    st.title("Risk Assessment")
    st.write("Risk assessment tools coming soon!")

def show_vehicle_comparison():
    st.title("Vehicle Comparison")
    st.write("Vehicle comparison tools coming soon!")

def show_maintenance_predictor():
    st.title("Maintenance Predictor")
    st.write("Maintenance prediction tools coming soon!")

def show_insurance_calculator():
    st.title("Insurance Calculator")
    st.write("Insurance calculator tools coming soon!")

# --- MAIN PAGE LOGIC ---
if selected_space == "Home":
    show_home_page()
elif selected_space == "Data Science Space":
    if selected_page == "Data Overview":
        show_data_overview()
    elif selected_page == "Data Statistics":
        show_data_statistics()
    elif selected_page == "Data Merging & Missingness":
        show_missingness_analysis()
    elif selected_page == "EDA":
        show_eda()
    elif selected_page == "Correlation Analysis":
        show_correlation_analysis()
    elif selected_page == "Category Analysis":
        show_category_analysis()
elif selected_space == "Production Space":
    if selected_page == "Risk Assessment":
        show_risk_assessment()
    elif selected_page == "Vehicle Comparison":
        show_vehicle_comparison()
    elif selected_page == "Maintenance Predictor":
        show_maintenance_predictor()
    elif selected_page == "Insurance Calculator":
        show_insurance_calculator()

# Stop if data loading failed
if df is None and selected_space != "Home":
    st.stop()
