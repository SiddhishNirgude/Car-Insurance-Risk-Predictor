import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer

# Set page configuration
st.set_page_config(
    page_title="TechImpact Solutions - Car Insurance Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
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
        final_integrated_df_cleaned = pd.read_csv("final_integrated_dataset.csv")
        return final_integrated_df_cleaned
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the CSV file is in the correct directory.")
        return None

# Load data
df = load_data()

# Main space selection in sidebar
st.sidebar.title("TechImpact Solutions")
space = st.sidebar.selectbox(
    "Select Space",
    ["Production Space", "Data Science Space"]
)

# Conditional navigation based on selected space
if space == "Production Space":
    production_pages = [
        "Risk Assessment Tool",
        "Vehicle Comparison",
        "Maintenance Predictor",
        "Insurance Calculator"
    ]
    selected_page = st.sidebar.selectbox("Navigate to", production_pages)
    
    # Production Space content (placeholder for now)
    st.title("🚗 Car Insurance Risk Assessment Tool")
    st.write("Production space features coming soon...")

else:  # Data Science Space
    data_science_pages = [
        "Introduction",
        "Data Overview",
        "Data Statistics",
        "Data Merging & Missingness",
        "EDA",
        "Correlation Analysis",
        "Category Analysis"
    ]
    selected_page = st.sidebar.selectbox("Navigate to", data_science_pages)

    # Data Science Space content
    if selected_page == "Introduction":
        st.title("🚗 Car Insurance Risk Predictor - Technical Analysis")
        
        # Project overview section
        st.header("Project Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            This technical analysis focuses on developing a sophisticated risk prediction system 
            for car insurance claims through three key aspects:
            
            1. **Insurance Claims Analysis**: Understanding patterns in insurance claims
            2. **Vehicle Safety Assessment**: Evaluating safety features and their impact
            3. **Maintenance Pattern Study**: Analyzing how maintenance affects risk
            """)
            
        with col2:
            st.image("Imageof-Auto-Insurance.jpg", caption="Car Insurance Analytics")

        # Data Sources section
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

        # Technical Methodology
        st.header("Technical Methodology")
        st.markdown("""
        ### Data Processing Pipeline:
        1. Data Integration and Cleaning
        2. Feature Engineering
        3. Model Development
        4. Performance Validation
        
        ### Key Technical Achievements:
        - Successful integration of three complex datasets
        - Development of advanced risk prediction models
        - Achievement of 96.04% prediction accuracy
        - Creation of interpretable feature importance metrics
        """)

    # Rest of your existing code for other data science pages...
    elif selected_page == "Data Overview":
        # Your existing Data Overview code
        st.title('Data Overview')
        # ... rest of the code

    elif selected_page == "Data Statistics":
        # Your existing Data Statistics code
        st.title('Data Statistics')
        # ... rest of the code

    elif selected_page == "Data Merging & Missingness":
        # Your existing Data Merging and Missingness code
        st.title('Data Merging and Missingness Analysis')
        # ... rest of the code

if df is None:
    st.stop()
