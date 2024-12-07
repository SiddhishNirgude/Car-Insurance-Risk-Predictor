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

# Utility function to load a dataset (cached)
@st.cache_data
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as e:
        return None

# Load datasets
car_insurance_claim = load_dataset("car_insurance_claim.csv")
vehicle_features_data = load_dataset("Vehicle features data.csv")
vehicle_maintenance_data = load_dataset("vehicle_maintenance_data.csv")

# Load merged dataset
merged_dataset = load_dataset("final_integrated_dataset.csv")

# Check for dataset loading errors and show relevant messages
if car_insurance_claim is None:
    st.error("Error: Car Insurance Claims Data could not be loaded. Please check the file path.")
if vehicle_features_data is None:
    st.error("Error: Vehicle Features Data could not be loaded. Please check the file path.")
if vehicle_maintenance_data is None:
    st.error("Error: Vehicle Maintenance Data could not be loaded. Please check the file path.")
if merged_dataset is None:
    st.error("Error: Merged dataset could not be loaded. Please check the file path.")

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


# --- DATA SCIENCE SPACE PAGES ---
def show_data_overview():
    st.title("Data Overview")
    
    # Dropdown for dataset selection
    dataset_option = st.selectbox("Select Dataset", ["Car Insurance Claims", "Vehicle Features Data", 
                                                   "Vehicle Maintenance Data", "Merged Dataset"])
    
    # Load the corresponding dataset based on the selection
    if dataset_option == "Car Insurance Claims":
        df = car_insurance_claim
    elif dataset_option == "Vehicle Features Data":
        df = vehicle_features_data
    elif dataset_option == "Vehicle Maintenance Data":
        df = vehicle_maintenance_data
    else:  # Merged Dataset
        df = merged_dataset

    # Display the dataset table
    if df is not None:
        st.write("### Dataset Snapshot:")
        st.dataframe(df.head())
        st.write(f"**Dataset Shape**: {df.shape}")
    else:
        st.error("Data not loaded. Please check the source file.")

    # Dropdown to select a column from the selected dataset
    if df is not None:
        column_option = st.selectbox("Select Column", df.columns)

        # Display column definition or description (you can customize this dictionary)
        column_definitions = {
            'Age': 'Age of the insured individual.',
            'Policy Type': 'Type of the insurance policy (e.g., Full Coverage, Liability).',
            'Claim Amount': 'Amount claimed by the insured individual for the incident.',
            # Add other columns here with their definitions...
        }
        
        # Show column definition
        if column_option in column_definitions:
            st.write(f"### Column Definition: {column_option}")
            st.write(column_definitions[column_option])
        else:
            st.write("No definition available for this column.")



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
