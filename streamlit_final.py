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
insurance_clean = load_dataset("insurance_clean.csv")
features_clean = load_dataset("features_clean.csv")
maintenance_clean = load_dataset("maintenance_clean.csv")

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
        ["Data Overview", "Data Statistics","Data Cleaning", "Data Merging & Missingness", 
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
            'ID': 'Unique identifier for each record in the dataset.',
        'KIDSDRIV': 'Indicates whether there are children in the household who drive (1: Yes, 0: No).',
        'BIRTH': 'Birthdate of the insured individual in the format (DDMMMYY).',
        'AGE': 'Age of the insured individual.',
        'HOMEKIDS': 'Indicates whether the individual has children at home (1: Yes, 0: No).',
        'YOJ': 'Years of job experience of the insured individual.',
        'INCOME': 'Annual income of the insured individual, formatted as a string (e.g., $67,349).',
        'PARENT1': 'Indicates whether the insured individual is a single parent (Yes/No).',
        'HOME_VAL': 'Estimated value of the individual’s home, formatted as a string (e.g., $0).',
        'MSTATUS': 'Marital status of the insured individual (e.g., Single, Married).',
        'GENDER': 'Gender of the insured individual (M: Male, F: Female).',
        'EDUCATION': 'Highest level of education attained by the insured individual (e.g., PhD, Bachelor’s).',
        'OCCUPATION': 'Occupation of the insured individual (e.g., Professional, Technician).',
        'TRAVTIME': 'Time taken (in minutes) by the insured individual to travel to work.',
        'CAR_USE': 'Purpose for which the car is used (e.g., Private, Business).',
        'BLUEBOOK': 'Blue Book value of the car, formatted as a string (e.g., $14,230).',
        'TIF': 'Time in years the individual has been with the current insurance company.',
        'CAR_TYPE': 'Type of the car (e.g., Minivan, Sedan, SUV).',
        'RED_CAR': 'Indicates whether the insured individual owns a red-colored car (yes/no).',
        'OLDCLAIM': 'Previous claim amount made by the insured individual, formatted as a string (e.g., $4,461).',
        'CLM_FREQ': 'Frequency of claims made by the insured individual.',
        'REVOKED': 'Indicates whether the insured individual’s policy has been revoked (Yes/No).',
        'MVR_PTS': 'Number of points on the insured individual’s motor vehicle record (MVR).',
        'CLM_AMT': 'Claim amount for the incident, formatted as a string (e.g., $0).',
        'CAR_AGE': 'Age of the car in years.',
        'CLAIM_FLAG': 'Indicates whether a claim was filed (1: Yes, 0: No).',
        'URBANICITY': 'Level of urbanization where the insured individual lives',
        'policy_id': 'Unique identifier for each policy in the dataset.',
        'policy_tenure': 'Duration of the insurance policy, represented as a fraction of time.',
        'age_of_car': 'Age of the car in years.',
        'age_of_policyholder': 'Age of the individual holding the insurance policy, represented as a fraction of time.',
        'area_cluster': 'Area cluster where the policyholder resides (e.g., C1, C2).',
        'population_density': 'Population density of the area where the policyholder resides.',
        'make': 'Brand or manufacturer of the vehicle (represented as an integer code).',
        'segment': 'Segment classification of the vehicle (e.g., A, B, C).',
        'model': 'Model of the vehicle (e.g., M1, M2).',
        'fuel_type': 'Type of fuel the vehicle uses (e.g., CNG, Petrol, Diesel).',
        'max_torque': 'Maximum torque of the vehicle’s engine, given in Nm and at a specified RPM (e.g., 60Nm@3500rpm).',
        'max_power': 'Maximum power of the vehicle’s engine, given in bhp and at a specified RPM (e.g., 40.36bhp@6000rpm).',
        'engine_type': 'Type of engine used in the vehicle (e.g., F8D Petrol Engine).',
        'airbags': 'Number of airbags installed in the vehicle.',
        'is_esc': 'Indicates whether the vehicle has Electronic Stability Control (Yes/No).',
        'is_adjustable_steering': 'Indicates whether the steering wheel is adjustable (Yes/No).',
        'is_tpms': 'Indicates whether the vehicle is equipped with a Tire Pressure Monitoring System (Yes/No).',
        'is_parking_sensors': 'Indicates whether the vehicle has parking sensors (Yes/No).',
        'is_parking_camera': 'Indicates whether the vehicle has a parking camera (Yes/No).',
        'rear_brakes_type': 'Type of rear brakes in the vehicle (e.g., Drum, Disc).',
        'displacement': 'Engine displacement of the vehicle in cubic centimeters (cc).',
        'cylinder': 'Number of cylinders in the engine of the vehicle.',
        'transmission_type': 'Type of transmission in the vehicle (e.g., Manual, Automatic).',
        'gear_box': 'Number of gears in the vehicle’s gearbox.',
        'steering_type': 'Type of steering in the vehicle (e.g., Power, Manual).',
        'turning_radius': 'Turning radius of the vehicle in meters.',
        'length': 'Length of the vehicle in millimeters.',
        'width': 'Width of the vehicle in millimeters.',
        'height': 'Height of the vehicle in millimeters.',
        'gross_weight': 'Gross weight of the vehicle in kilograms.',
        'is_front_fog_lights': 'Indicates whether the vehicle is equipped with front fog lights (Yes/No).',
        'is_rear_window_wiper': 'Indicates whether the vehicle has a rear window wiper (Yes/No).',
        'is_rear_window_washer': 'Indicates whether the vehicle has a rear window washer (Yes/No).',
        'is_rear_window_defogger': 'Indicates whether the vehicle has a rear window defogger (Yes/No).',
        'is_brake_assist': 'Indicates whether the vehicle is equipped with brake assist (Yes/No).',
        'is_power_door_locks': 'Indicates whether the vehicle has power door locks (Yes/No).',
        'is_central_locking': 'Indicates whether the vehicle has central locking (Yes/No).',
        'is_power_steering': 'Indicates whether the vehicle has power steering (Yes/No).',
        'is_driver_seat_height_adjustable': 'Indicates whether the driver’s seat is height adjustable (Yes/No).',
        'is_day_night_rear_view_mirror': 'Indicates whether the vehicle has a day/night rear view mirror (Yes/No).',
        'is_ecw': 'Indicates whether the vehicle has an electronic control unit (Yes/No).',
        'is_speed_alert': 'Indicates whether the vehicle is equipped with a speed alert system (Yes/No).',
        'ncap_rating': 'Safety rating of the vehicle as per NCAP (e.g., 0, 4).',
        'is_claim': 'Indicates whether the vehicle has been involved in a claim (1: Yes, 0: No).',
        'Vehicle_Model': 'Model of the vehicle (e.g., Truck, Sedan, SUV).',
        'Mileage': 'Total distance traveled by the vehicle, recorded in kilometers or miles.',
        'Maintenance_History': 'Condition of the vehicle’s maintenance history (e.g., Good, Fair, Poor).',
        'Reported_Issues': 'Number of issues reported for the vehicle.',
        'Vehicle_Age': 'Age of the vehicle in years.',
        'Fuel_Type': 'Type of fuel used by the vehicle (e.g., Electric, Petrol, Diesel).',
        'Transmission_Type': 'Type of transmission in the vehicle (e.g., Automatic, Manual).',
        'Engine_Size': 'Size of the vehicle’s engine in cubic centimeters (cc).',
        'Odometer_Reading': 'Current odometer reading of the vehicle in kilometers or miles.',
        'Last_Service_Date': 'Date when the vehicle was last serviced.',
        'Warranty_Expiry_Date': 'Date when the vehicle’s warranty expires.',
        'Owner_Type': 'Type of vehicle ownership (e.g., First, Second, Third).',
        'Insurance_Premium': 'Amount paid for the vehicle’s insurance premium.',
        'Service_History': 'Number of times the vehicle has been serviced.',
        'Accident_History': 'Number of accidents the vehicle has been involved in.',
        'Fuel_Efficiency': 'Fuel efficiency of the vehicle, typically in kilometers per liter or miles per gallon.',
        'Tire_Condition': 'Condition of the vehicle’s tires (e.g., New, Worn, Damaged).',
        'Brake_Condition': 'Condition of the vehicle’s brakes (e.g., New, Worn, Damaged).',
        'Battery_Status': 'Status of the vehicle’s battery (e.g., Weak, Strong, Good).',
        'Need_Maintenance': 'Indicates whether the vehicle requires maintenance (1: Yes, 0: No).'
        }
        
        # Show column definition
        if column_option in column_definitions:
            st.write(f"### Column Definition: {column_option}")
            st.write(column_definitions[column_option])
        else:
            st.write("No definition available for this column.")



def show_data_statistics():
    st.title("Data Statistics")
    
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

    if df is not None:
        # Show the dataset table
        st.write("### Data Table:")
        st.dataframe(df)
        
        # Show the data types table
        st.write("### Data Types:")
        st.dataframe(df.dtypes)
        
        # Show descriptive statistics for numeric features
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        st.write("### Descriptive Statistics for Numeric Features:")
        st.dataframe(df[numeric_cols].describe())
    else:
        st.error("Data not loaded. Please check the source file.")

def show_data_cleaning_steps():
    st.title("Data Cleaning Steps")
    
    # Dropdown for dataset selection
    dataset_option = st.selectbox("Select Dataset", ["Car Insurance Claims", "Vehicle Features Data", 
                                                   "Vehicle Maintenance Data", "Merged Dataset"])

    # Check for dataset loading errors
    if insurance_clean is None:
        st.error("Error: Car Insurance Claims Data could not be loaded. Please check the file path.")
    if features_clean is None:
        st.error("Error: Vehicle Features Data could not be loaded. Please check the file path.")
    if maintenance_clean is None:
        st.error("Error: Vehicle Maintenance Data could not be loaded. Please check the file path.")
    if merged_dataset is None:
        st.error("Error: Merged dataset could not be loaded. Please check the file path.")
    
    # Show the cleaning steps for the corresponding dataset
    if dataset_option == "Car Insurance Claims":
        st.write("""
        ### Car Insurance Claims Data Cleaning Steps:
        
        1. **Remove Duplicates**:
            - Duplicates were removed to ensure unique entries in the dataset.
        
        2. **Clean Monetary Columns**:
            - The `INCOME`, `HOME_VAL`, `BLUEBOOK`, `OLDCLAIM`, and `CLM_AMT` columns were cleaned by removing dollar signs (`$`) and commas, and then converting the values to numeric types.
        
        3. **Convert `BIRTH` to Datetime**:
            - The `BIRTH` column was converted into a proper datetime format (`%d%b%y`).
        
        4. **Remove `z_` Prefix in Categorical Variables**:
            - Categorical variables such as `MSTATUS`, `GENDER`, `CAR_TYPE`, `URBANICITY` had the `z_` prefix removed to standardize the data.
        
        5. **Standardize Binary Columns**:
            - Columns such as `PARENT1`, `RED_CAR`, and `REVOKED` were mapped to binary values (`1` for `Yes` and `0` for `No`).
        """)
        
        # Example of data (if loaded)
        st.write("### Cleaned Data Sample:")
        st.write(insurance_clean.head())  # Assuming `insurance_clean` is already loaded

    elif dataset_option == "Vehicle Features Data":
        st.write("""
        ### Vehicle Features Data Cleaning Steps:
        
        1. **Convert Yes/No Columns to Binary**:
            - Columns like `is_esc`, `is_tpms`, `is_parking_sensors`, etc., were converted to `1` for `Yes` and `0` for `No`.
        
        2. **Clean `area_cluster` Column**:
            - Removed extra spaces from the `area_cluster` column to ensure consistent formatting.
        
        3. **Standardize Categorical Columns**:
            - Categorical columns like `fuel_type`, `segment`, `model`, `engine_type`, etc., were stripped of leading/trailing spaces to standardize the data.
        
        4. **Ensure `policy_id` is a String**:
            - The `policy_id` column was converted to a string type to ensure uniformity.
        
        5. **Ensure Numeric Columns are Properly Typed**:
            - Columns like `policy_tenure`, `age_of_car`, and `age_of_policyholder` were converted to numeric types.
        """)
        
        # Example of data (if loaded)
        st.write("### Cleaned Data Sample:")
        st.write(features_clean.head())  # Assuming `features_clean` is already loaded

    elif dataset_option == "Vehicle Maintenance Data":
        st.write("""
        ### Vehicle Maintenance Data Cleaning Steps:
        
        1. **Convert Date Columns to Datetime**:
            - The `Last_Service_Date` and `Warranty_Expiry_Date` columns were converted into proper datetime format.
        
        2. **Standardize Categorical Columns**:
            - The `Vehicle_Model`, `Fuel_Type`, and `Transmission_Type` columns were stripped of spaces to ensure uniformity.
        
        3. **Encode Categorical Columns**:
            - Categorical columns like `Maintenance_History`, `Owner_Type`, `Tire_Condition`, and `Brake_Condition` were encoded with numerical codes.
        
        4. **Battery Status**:
            - The `Battery_Status` column was mapped to `0` for `Weak` and `1` for `New`.
        
        5. **Ensure Numeric Columns are Properly Typed**:
            - Columns like `Mileage`, `Reported_Issues`, `Vehicle_Age`, etc., were converted to numeric types.
        """)
        
        # Example of data (if loaded)
        st.write("### Cleaned Data Sample:")
        st.write(maintenance_clean.head())  # Assuming `maintenance_clean` is already loaded

    else:  # Merged Dataset
        st.write("""
        ### Merged Dataset Data Cleaning Steps:
        
        1. **Merge Multiple Datasets**:
            - The `Car Insurance Claims`, `Vehicle Features`, and `Vehicle Maintenance` datasets were merged on the `policy_id` column to create a unified dataset.
        
        2. **Handle Missing Values**:
            - Missing values in critical columns were filled with appropriate values or dropped based on the significance of the column.
        
        3. **Remove Outliers**:
            - Extreme values that seemed unrealistic were identified and removed.
        
        4. **Standardize Categorical Variables**:
            - Categorical variables across the merged dataset were standardized to ensure uniformity in representation.
        """)
        
        # Example of data (if loaded)
        st.write("### Cleaned Data Sample:")
        st.write(merged_dataset.head())

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
    elif selected_page == "Data Cleaning":
        show_data_cleaning_steps()
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
