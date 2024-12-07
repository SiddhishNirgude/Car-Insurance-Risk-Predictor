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

# Function to show introduction content (will be used both in landing and as a page)
def show_introduction_content():
    [Your existing show_introduction_content() function code remains the same]

# Load data (cached)
@st.cache_data
def load_data():
    [Your existing load_data() function code remains the same]

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

# Second level navigation based on selected space
if selected_space == "Data Science Space":
    selected_page = st.sidebar.selectbox(
        "Select Page",
        ["Introduction",
         "Data Overview",
         "Data Statistics",
         "Data Merging & Missingness",
         "EDA",
         "Correlation Analysis",
         "Category Analysis"]
    )
    
    # Clear the main area before showing new content
    st.empty()
    
    # Show content based on selected page
    if selected_page == "Introduction":
        show_introduction_content()
        
    elif selected_page == "Data Overview":
        st.title("Data Overview")
        st.header("Dataset Description")
        if df is not None:
            st.dataframe(df.head())
            st.write(f"Dataset Shape: {df.shape}")
            
    elif selected_page == "Data Statistics":
        st.title("Data Statistics")
        if df is not None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            st.header("Numeric Features Statistics")
            st.dataframe(df[numeric_cols].describe())
            
    elif selected_page == "Data Merging & Missingness":
        st.title("Data Merging and Missingness Analysis")
        if df is not None:
            missing = df.isnull().sum()
            st.write("Missing Values Summary:")
            st.write(missing[missing > 0])
            
    # Add other pages as needed

elif selected_space == "Production Space":
    selected_page = st.sidebar.selectbox(
        "Select Page",
        ["Risk Assessment",
         "Vehicle Comparison",
         "Maintenance Predictor",
         "Insurance Calculator"]
    )
    
    # Clear the main area
    st.empty()
    st.info("Production Space features are under development.")

# Stop if data loading failed
if df is None:
    st.stop()
