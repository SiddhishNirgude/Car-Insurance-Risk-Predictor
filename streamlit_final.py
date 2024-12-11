import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        # Check if the file is a URL or local path
        if file_path.startswith('http'):
            data = pd.read_csv(file_path)  # If it's a URL, load it directly
        else:
            # Local file loading
            data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as e:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# GitHub raw URLs for the datasets
balanced_data_url = "https://raw.githubusercontent.com/SiddhishNirgude/Car-Insurance-Risk-Predictor/refs/heads/main/final_integrated_df_cleaned_balanced_3.csv"

# Load datasets using URLs (adjusted to GitHub URLs)
balanced_data = load_dataset(balanced_data_url)
# Load datasets
car_insurance_claim = load_dataset("car_insurance_claim.csv")
vehicle_features_data = load_dataset("Vehicle features data.csv")
vehicle_maintenance_data = load_dataset("vehicle_maintenance_data.csv")
insurance_clean = load_dataset("insurance_clean.csv")
features_clean = load_dataset("features_clean.csv")
maintenance_clean = load_dataset("maintenance_clean.csv")
insurance_after_imputation = load_dataset("insurance_after_imputation.csv")
features_after_imputation = load_dataset("features_after_imputation.csv")
features_induced_missing = load_dataset("features_missing.csv")
maintenance_after_imputation = load_dataset("maintenance_after_imputation.csv")
merged_dataset = load_dataset("final_integrated_dataset.csv")
merged_reduced_cols_dataset = load_dataset("final_integrated_df_cleaned.csv")

# Add the new datasets
insurance_encoded = load_dataset("insurance_encoded.csv")
features_encoded = load_dataset("features_encoded.csv")
maintenance_encoded = load_dataset("maintenance_encoded.csv")



# Check for dataset loading errors and show relevant messages
if car_insurance_claim is None:
    st.error("Error: Car Insurance Claims Data could not be loaded. Please check the file path.")
if vehicle_features_data is None:
    st.error("Error: Vehicle Features Data could not be loaded. Please check the file path.")
if vehicle_maintenance_data is None:
    st.error("Error: Vehicle Maintenance Data could not be loaded. Please check the file path.")
if insurance_clean is None:
    st.error("Error: Cleaned Car Insurance Claims Data could not be loaded. Please check the file path.")
if features_clean is None:
    st.error("Error: Cleaned Vehicle Features Data could not be loaded. Please check the file path.")
if maintenance_clean is None:
    st.error("Error: Cleaned Vehicle Maintenance Data could not be loaded. Please check the file path.")
if insurance_after_imputation is None:
    st.error("Error: After Imputation Car Insurance Claims Data could not be loaded. Please check the file path.")
if features_after_imputation is None:
    st.error("Error: After Imputation Vehicle Features Data could not be loaded. Please check the file path.")
if features_induced_missing is None:
    st.error("Error: After Imputation Vehicle Features Data could not be loaded. Please check the file path.")
if maintenance_after_imputation is None:
    st.error("Error: After Imputation Vehicle Maintenance Data could not be loaded. Please check the file path.")
if merged_dataset is None:
    st.error("Error: Merged dataset could not be loaded. Please check the file path.")
if insurance_encoded is None:
    st.error("Error: Encoded Insurance Data could not be loaded. Please check the file path.")
if features_encoded is None:
    st.error("Error: Encoded Features Data could not be loaded. Please check the file path.")
if maintenance_encoded is None:
    st.error("Error: Encoded Maintenance Data could not be loaded. Please check the file path.")
if balanced_data is None:
    st.error("Error:  Data could not be loaded. Please check the file path.")
if merged_reduced_cols_dataset is None:
    st.error("Error:  Data could not be loaded. Please check the file path.")

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
        ["Data Overview", "Data Statistics","Data Cleaning", "Data Missingness Analysis", "Data Transformations", "Data Merging & Integration",
         "EDA", "Correlation Analysis", "Dimensionality Reduction"]
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
        st.image("images/Imageof-Auto-Insurance.jpg", caption="Car Insurance Analytics")

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
    st.title("Data Missingness Analysis")

    # Dropdown to select the dataset
    dataset_option = st.selectbox(
        "Select Dataset for Missingness Analysis",
        ["Car Insurance Claims", 
         "Vehicle Features Data", 
         "Vehicle Maintenance Data"]
    )

    # Map the selection to datasets

    if dataset_option == "Car Insurance Claims":
        df_before = insurance_clean
        df_after = insurance_after_imputation
        imputation_steps = """
        1. **Identified Missing Values**: Focused on missing values in `AGE`, `OCCUPATION`, and key numerical features like `YOJ`, `INCOME`, `HOME_VAL`, and `CAR_AGE`.
        2. **Imputation Strategies**:
            - **Median Imputation**: Used for `AGE`, as it minimizes distortion caused by outliers.
            - **Mode Imputation**: Applied to `OCCUPATION`, replacing missing values with the most frequent category.
            - **K-Nearest Neighbors (KNN) Imputation**: Utilized for numerical features such as `YOJ`, `INCOME`, `HOME_VAL`, and `CAR_AGE`, imputing missing values based on similarities in other data points.
        3. **Rechecked Missingness**: Verified after imputation that all missing values were addressed, ensuring no residual missingness.
        """
    elif dataset_option == "Vehicle Features Data":
        df_before = vehicle_features_data
        df_induced = features_induced_missing
        df_after = features_after_imputation
        imputation_steps = """
        1. **Induced Missingness**: Introduced artificial missingness in selected columns (`age_of_car`, `age_of_policyholder`, `population_density`, `displacement`, `turning_radius`, `make`, `segment`, `fuel_type`) to simulate real-world scenarios.
        2. **Imputation Strategies**:
            - **Mode Imputation**: Applied to categorical features (`segment`, `fuel_type`) by replacing missing values with the most frequent category.
            - **K-Nearest Neighbors (KNN) Imputation**: Used for numerical features (`age_of_car`, `age_of_policyholder`, `population_density`, `displacement`, `turning_radius`, `make`) by leveraging similarities among other features to fill missing values.
        3. **Rechecked Missingness**: Verified after imputation that all missing values were addressed, ensuring the dataset's completeness.
        """

    elif dataset_option == "Vehicle Maintenance Data":
        df_before = maintenance_clean
        df_after = maintenance_after_imputation
        imputation_steps = """
        1. **Identified Missing Values**: Focused on missing maintenance logs, repair dates, and costs.
        2. **Imputation Strategies**:
            - **Forward Fill**: Used for sequential maintenance records to carry forward the previous values for date and cost.
            - **Mean Imputation**: Applied for numerical values like maintenance cost, replacing missing data with the mean of the column.
        3. **Rechecked Missingness**: After imputation, confirmed no missing data remained in critical columns.
        """
    else:
        st.error("Invalid selection.")
        return

    # Display imputation steps above heatmaps
    st.write("### Steps Taken for Removing Missingness (Imputation):")
    st.write(imputation_steps)

    # Show missingness heatmaps for the selected dataset
    if dataset_option == "Vehicle Features Data":
        # Original Data (Before Inducing Missingness)
        st.write("### Missingness Heatmap for Vehicle Features Data (Before Inducing Missingness):")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_before.isnull(), cbar=False, cmap="viridis")
        st.pyplot(fig)

        # Induced Missingness Data
        st.write("### Missingness Heatmap for Vehicle Features Data (After Inducing Missingness):")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_induced.isnull(), cbar=False, cmap="viridis")
        st.pyplot(fig)

        # After Imputation
        st.write("### Missingness Heatmap for Vehicle Features Data (After Imputation):")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_after.isnull(), cbar=False, cmap="viridis")
        st.pyplot(fig)
    else:
        # Heatmap for before imputation
        if df_before is not None:
            st.write(f"### Missingness Heatmap for {dataset_option} (Before Imputation):")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_before.isnull(), cbar=False, cmap="viridis")
            st.pyplot(fig)

        # Heatmap for after imputation
        if df_after is not None:
            st.write(f"### Missingness Heatmap for {dataset_option} (After Imputation):")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_after.isnull(), cbar=False, cmap="viridis")
            st.pyplot(fig)

def show_data_transformations():
    """
    Display data transformations steps and outputs for selected datasets.
    """
    # Dropdown for dataset selection
    dataset_option = st.selectbox(
        "Select Dataset for Transformations",
        ["Car Insurance Claims", 
         "Vehicle Features Data", 
         "Vehicle Maintenance Data"]
    )

    # Map dataset options to datasets and transformations documentation
    if dataset_option == "Car Insurance Claims":
        dataset = insurance_encoded
        transformation_steps = """
        ### Data Transformation Steps for Insurance Encoded Dataset:
        
        1. **Binary Encoding**:
            - Applied Label Encoding to categorical columns such as `MSTATUS`, `GENDER`, `CAR_USE`, and `URBANICITY` to convert them into binary (0 and 1) values.
        
        2. **Ordinal Encoding for Education**:
            - Applied Ordinal Encoding to the `EDUCATION` column with the order `['<High School', 'z_High School', 'Bachelors', 'Masters', 'PhD']` to assign numeric values based on education levels.
        
        3. **Ordinal Encoding for Occupation**:
            - Applied Ordinal Encoding to the `OCCUPATION` column with the order `['Student', 'Home Maker', 'z_Blue Collar', 'Clerical', 'Professional', 'Manager', 'Lawyer', 'Doctor']` to convert occupation levels into numeric values.
        
        4. **One-Hot Encoding for CAR_TYPE**:
            - Applied One-Hot Encoding to the `CAR_TYPE` column to create separate binary columns for each car type (e.g., `CAR_TYPE_Panel Truck`, `CAR_TYPE_Pickup`, etc.).
        
        5. **Scaling Numerical Variables**:
            - Applied Standard Scaling (Z-Score normalization) to numerical columns such as `AGE`, `YOJ`, `INCOME`, `HOME_VAL`, `TRAVTIME`, `BLUEBOOK`, `OLDCLAIM`, `CLM_AMT`, and `CAR_AGE` to ensure they are on the same scale with a mean of 0 and standard deviation of 1.
        
        6. **Removing Outliers**:
            - Removed outliers using a Z-score threshold of 3 for numerical columns (`AGE`, `YOJ`, `INCOME`, `HOME_VAL`, `TRAVTIME`, `BLUEBOOK`, `OLDCLAIM`, `CLM_AMT`, and `CAR_AGE`).
        """

    elif dataset_option == "Vehicle Features Data":
        dataset = features_encoded
        transformation_steps = """
        ### Data Transformation Steps for Features Encoded Dataset:
        1. **Drop High Cardinality Columns**:
            - Columns like `model`, `engine_type`, and `area_cluster` were dropped to reduce dimensionality.
        2. **One-Hot Encoding**:
            - Applied to nominal categorical columns such as `SEGMENT`, `FUEL_TYPE`, `TRANSMISSION_TYPE`, `STEERING_TYPE`, and `REAR_BRAKES_TYPE`.
        3. **Binary Columns**:
            - Binary columns such as `is_esc`, `is_parking_sensors`, and `is_claim` were retained as-is since they are already encoded.
        4. **Feature Scaling**:
            - Standard Scaling was applied to numerical columns like `POLICY_TENURE`, `AGE_OF_CAR`, and `GROSS_WEIGHT` to normalize their values.
        5. **Outlier Removal**:
            - Outliers were removed from numeric columns using a Z-score threshold of 3 to ensure the model is not affected by extreme values.
        """
    
    elif dataset_option == "Vehicle Maintenance Data":
        dataset = maintenance_encoded
        transformation_steps = """
        ### Data Transformation Steps for Vehicle Maintenance Dataset:
        1. **Fix Battery Status Encoding**:
            - The `Battery_Status` column is ordinal and is encoded using `OrdinalEncoder`, with categories ordered as `['Weak', 'Good', 'New']`. This assigns integer values: `Weak = 0`, `Good = 1`, and `New = 2`.
        2. **Drop Original Categorical Columns**:
            - The original categorical columns that have been encoded or are not needed for further analysis are dropped. These include:
              - `Maintenance_History`, `Owner_Type`, `Tire_Condition`, `Brake_Condition`, `Battery_Status`.
        3. **One-Hot Encoding for Nominal Categorical Variables**:
            - One-hot encoding is applied to nominal categorical columns such as:
              - `Vehicle_Model`, `Fuel_Type`, `Transmission_Type`.
            - This creates binary columns for each category within these columns.
        4. **Transform Date Columns to Numeric Features**:
            - The `Last_Service_Date` and `Warranty_Expiry_Date` columns are transformed into numerical features:
              - `days_since_service`: The number of days since the last service date relative to a fixed date (`2024-03-01`).
              - `days_until_warranty_expires`: The number of days until the warranty expires relative to the fixed date.
            - The original date columns are dropped after transformation.
        5. **Scale Numeric Columns**:
            - Standard scaling is applied to the following numeric columns:
              - `Mileage`, `Reported_Issues`, `Vehicle_Age`, `Engine_Size`, `Odometer_Reading`, `Insurance_Premium`, `Service_History`, `Accident_History`, `Fuel_Efficiency`, `days_since_service`, `days_until_warranty_expires`.
            - This scales the values of these columns to have zero mean and unit variance.
        6. **Remove Outliers (Z-Score Threshold)**:
            - Outliers are removed from numeric columns using a Z-score threshold of 3. Any rows with Z-scores greater than 3 for any numeric column are excluded from the dataset.
        7. **Keep Coded Columns As Is**:
            - Coded columns such as `Maintenance_History_Code`, `Owner_Type_Code`, `Tire_Condition_Code`, `Brake_Condition_Code`, and `Battery_Status_Code` are retained as-is.
        """
    
    # Display transformation steps documentation
    st.write(transformation_steps)

    # Display the transformed dataset
    st.write("### Transformed Dataset:")
    st.dataframe(dataset)


def show_data_merging():
    """
    Displays the steps involved in data integration, cleaning, and transformation with Streamlit tabs.
    """

    # Create tabs for each section
    tab1, tab2, tab3 = st.tabs(["Data Integration", "Cleaning After Integration", "Data Transformation"])

    # Data Integration Tab
    with tab1:
        st.header("Steps for Data Integration")
        st.markdown("""
        **Steps:**
        1. **Adding Unique Identifiers (`policy_id_no`)**:
            - Unique IDs were added to ensure datasets could be merged accurately.
        2. **Merging Datasets**:
            - Insurance, features, and maintenance datasets were merged using `policy_id_no` as the key.
        3. **Final Integrated Dataset**:
            - Only rows with matching `policy_id_no` across all datasets were retained.
        """)
        # Assuming `merged_dataset` is already loaded
        st.subheader("Final Integrated Dataset")
        st.dataframe(merged_dataset.head())
        st.write("Shape of the final integrated dataset:", merged_dataset.shape)

    # Cleaning After Integration Tab
    with tab2:
        st.header("Steps for Cleaning After Integration")
        st.markdown("""
        **Steps:**
        1. **Removing Duplicates**:
            - Duplicate rows were removed to ensure data consistency.
        2. **Checking for Missing Values**:
            - Missing data analysis was performed, identifying columns with null values.
        3. **Dropping Unnecessary Columns**:
            - Irrelevant columns were removed to streamline the dataset.
        """)

        # Cleaning: Removing duplicates
        merged_dataset_no_duplicates = merged_dataset.drop_duplicates()
        st.subheader("Cleaned Dataset (After Removing Duplicates)")
        st.write("Shape after removing duplicates:", merged_dataset_no_duplicates.shape)
        st.dataframe(merged_dataset_no_duplicates.head())

        # Columns to drop
        columns_to_drop = [
            'ID', 'BLUEBOOK', 'RED_CAR', 'policy_id', 'policy_tenure', 
            'age_of_car', 'age_of_policyholder', 'population_density', 
            'make', 'Vehicle_Model_Car', 'Vehicle_Model_Motorcycle', 
            'Vehicle_Model_SUV', 'Vehicle_Model_Truck', 'Vehicle_Model_Van', 
            'Fuel_Type_Electric', 'Fuel_Type_Petrol', 'Transmission_Type_Manual', 
            'days_since_service', 'days_until_warranty_expires'
        ]
        merged_dataset_cleaned = merged_dataset_no_duplicates.drop(columns=columns_to_drop)

        # Displaying datasets after dropping unnecessary columns
        st.subheader("Cleaned Dataset (After Dropping Unnecessary Columns)")
        st.write("Shape of the cleaned dataset:", merged_dataset_cleaned.shape)
        st.dataframe(merged_dataset_cleaned.head())
        st.write("Columns dropped during cleaning:")
        st.write(columns_to_drop)

    # Data Transformation Tab
    with tab3:
        st.header("Data Transformation After Integration")
        st.markdown("""
        **Steps:**
        1. **Check Class Imbalance**:
            - We analyzed the class distribution of target variables before SMOTE balancing
            - Initial analysis showed significant imbalance in CLAIM_FLAG (73.3% vs 26.7%)
            - Severe imbalance in Need_Maintenance (80.4% vs 19.6%)
            - Extreme imbalance in is_claim (94.1% vs 5.9%)
            
        2. **Initial SMOTE Balancing Attempt**:
            - Applied SMOTE to all target variables simultaneously
            - Carefully considered the balancing ratio for each variable
            - Aimed for optimal balance while preserving data quality
            
        3. **Optimized SMOTE Application**:
            - Implemented variable-specific balancing strategies
            - CLAIM_FLAG: Applied complete balancing (50-50 split)
            - Need_Maintenance: Near-balanced distribution (53.1% vs 46.9%)
            - is_claim: Moderate balancing (62.3% vs 37.7%)
            
        4. **Final Dataset Optimization**:
            - Each target variable achieved its optimal balance ratio
            - CLAIM_FLAG achieved perfect balance for maximum learning effectiveness
            - Need_Maintenance and is_claim maintained controlled imbalance to prevent overfitting
            - Preserved data quality while improving minority class representation
            
        **Impact on Model Training**:
            - Differentiated balancing strategies improve model robustness
            - Perfect balance in CLAIM_FLAG ensures unbiased prediction of insurance claims
            - Near-balance in maintenance prediction enables better preventive insights
            - Controlled balance in is_claim reduces bias while preventing synthetic data dominance
        """)


        # List of columns to balance
        columns_to_balance = ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']
        
        # Verify columns exist in both datasets
        available_columns = [col for col in columns_to_balance 
                           if col in merged_dataset_no_duplicates.columns 
                           and col in balanced_data.columns]

        if not available_columns:
            st.error("Target columns not found in datasets. Please check column names.")
            st.write("Available columns in original dataset:", merged_dataset_no_duplicates.columns.tolist())
            st.write("Available columns in balanced dataset:", balanced_data.columns.tolist())
            return
        
        for col in available_columns:
            st.subheader(f"Class Distribution Analysis for {col}")
            
            # Create two columns for side-by-side visualization
            col1, col2 = st.columns(2)
            
            try:
                # Analyzing class imbalance
                class_distribution_before = merged_dataset_no_duplicates[col].value_counts()
                class_distribution_after = balanced_data[col].value_counts()
                
                # Plot in first column
                with col1:
                    st.write("Before SMOTE")
                    fig_before = plot_pie_chart(class_distribution_before)
                    plt.figure(figsize=(6, 4))  # Reduce figure size
                    st.pyplot(fig_before)
                    
                # Plot in second column
                with col2:
                    st.write("After SMOTE")
                    fig_after = plot_pie_chart(class_distribution_after)
                    plt.figure(figsize=(6, 4))  # Reduce figure size
                    st.pyplot(fig_after)
                
                # Add remarks based on the variable
                if col == 'CLAIM_FLAG':
                    st.markdown("""
                    **Observations for Insurance Claims:**
                    - Initial distribution showed significant imbalance (73.3% vs 26.7%)
                    - After SMOTE: Achieved perfect balance (50.0% vs 50.0%)
                    - Complete balancing applied to maximize learning from both classes
                    """)
                
                elif col == 'Need_Maintenance':
                    st.markdown("""
                    **Observations for Maintenance Needs:**
                    - Severe initial imbalance (80.4% vs 19.6% needs maintenance)
                    - Post-SMOTE distribution improved to 53.1% vs 46.9%
                    - Near-balanced distribution achieved while maintaining data integrity
                    """)
                
                elif col == 'is_claim':
                    st.markdown("""
                    **Observations for Claim Status:**
                    - Highly skewed initial distribution (94.1% vs 5.9%)
                    - After SMOTE: Improved to 62.3% vs 37.7%
                    - Moderate balancing applied to reduce extreme imbalance while preventing overfitting
                    """)
                
                # Display counts table before and after SMOTE
                st.subheader(f"Distribution Counts for {col}")
                before_after_counts = pd.DataFrame({
                    'Before SMOTE': class_distribution_before,
                    'After SMOTE': class_distribution_after
                }).fillna(0).astype(int)
                st.dataframe(before_after_counts)
                
            except Exception as e:
                st.error(f"Error processing column {col}: {str(e)}")
            
            st.markdown("---")  # Add separator between variables

# Modified pie chart function with smaller size
def plot_pie_chart(class_distribution):
    fig, ax = plt.subplots(figsize=(5, 5))  # Reduced figure size
    ax.pie(class_distribution, labels=class_distribution.index, 
           autopct='%1.1f%%', startangle=90,
           colors=['#2E86C1', '#F39C12'])  # Consistent colors
    ax.axis('equal')
    return fig
        
### A. For Customer Demographics tab
# Add this function at the start of your show_eda function
def clean_categorical_columns(df):
    """
    Clean specific categorical columns by rounding and converting to int
    """
    df_cleaned = df.copy()
    
    # Clean EDUCATION
    df_cleaned['EDUCATION'] = df_cleaned['EDUCATION'].round().astype(int)
    # Map education codes to labels
    education_map = {
        0: 'High School',
        1: 'Bachelors',
        2: 'Masters',
        3: 'PhD',
        4: 'Other'
    }
    df_cleaned['EDUCATION'] = df_cleaned['EDUCATION'].map(education_map)

    # Clean OCCUPATION
    df_cleaned['OCCUPATION'] = df_cleaned['OCCUPATION'].round().astype(int)
    # Map occupation codes to labels
    occupation_map = {
        0: 'Blue Collar',
        1: 'Clerical',
        2: 'Professional',
        3: 'Manager',
        4: 'Home Maker',
        5: 'Student',
        6: 'Other'
    }
    df_cleaned['OCCUPATION'] = df_cleaned['OCCUPATION'].map(occupation_map)

    # Clean Battery_Status_Code
    df_cleaned['Battery_Status_Code'] = df_cleaned['Battery_Status_Code'].round().astype(int)
    # Map battery status codes
    battery_map = {
        0: 'Good',
        1: 'Fair',
        2: 'Poor'
    }
    df_cleaned['Battery_Status_Code'] = df_cleaned['Battery_Status_Code'].map(battery_map)

    return df_cleaned


### C. For risk_indicators tab
def process_risk_indicators(df):
    """
    Process risk indicator columns to map encoded values back to meaningful categories.
    
    Args:
        df: DataFrame containing the risk indicator columns
    
    Returns:
        DataFrame with processed risk indicators
    """
    df_processed = df.copy()
    
    # 1. REVOKED mapping (0/1 to No/Yes)
    df_processed['REVOKED_CAT'] = df_processed['REVOKED'].map({
        0: 'Not Revoked',
        1: 'Revoked'
    })
    
    # 2. CAR_USE mapping (0/1 to Private/Commercial)
    df_processed['CAR_USE_CAT'] = df_processed['CAR_USE'].map({
        0: 'Private',
        1: 'Commercial'
    })
    
    # 3. URBANICITY mapping (0/1 to Low/High)
    df_processed['URBANICITY_CAT'] = df_processed['URBANICITY'].map({
        0: 'Low Urbanicity',
        1: 'High Urbanicity'
    })
    
    # 4. Accident_History mapping (float ranges to severity categories)
    def map_accident_history(value):
        if value <= 1:
            return 'No Accidents'
        elif value <= 2:
            return 'Minor Accidents'
        elif value <= 3:
            return 'Moderate Accidents'
        else:
            return 'Severe Accidents'
            
    df_processed['ACCIDENT_HISTORY_CAT'] = df_processed['Accident_History'].apply(map_accident_history)
    
    # Create a color mapping dictionary for accident history categories
    accident_colors = {
        'No Accidents': 'green',
        'Minor Accidents': 'yellow',
        'Moderate Accidents': 'orange',
        'Severe Accidents': 'red'
    }
    
    return df_processed, accident_colors


def get_risk_category_descriptions():
    """
    Returns descriptions for each risk category for documentation purposes.
    """
    return {
        'REVOKED': {
            'Not Revoked': 'Driver license has never been revoked',
            'Revoked': 'Driver license has been revoked in the past'
        },
        'CAR_USE': {
            'Private': 'Vehicle used for personal/family purposes',
            'Commercial': 'Vehicle used for business/commercial purposes'
        },
        'URBANICITY': {
            'Low Urbanicity': 'Rural or less densely populated area',
            'High Urbanicity': 'Urban or densely populated area'
        },
        'ACCIDENT_HISTORY': {
            'No Accidents': 'No significant accident history (Score ≤ 1)',
            'Minor Accidents': 'Minor accidents or incidents (1 < Score ≤ 2)',
            'Moderate Accidents': 'Moderate accident history (2 < Score ≤ 3)',
            'Severe Accidents': 'Severe or frequent accidents (Score > 3)'
        }
    }


### D. For maintenance_metric tab
# 1. Base helper function for scaled values
def group_scaled_values(value, ranges, labels):
    """
    Helper function to group scaled values into meaningful categories
    
    Args:
        value: The scaled value to categorize
        ranges: List of range boundaries
        labels: List of labels for each range
    
    Returns:
        Appropriate label for the value
    """
    for i, r in enumerate(ranges[:-1]):
        if value <= r:
            return labels[i]
    return labels[-1]

# 2. Safe mapping helper function
def safe_map(series, mapping):
    """
    Safely map values to categories, handling both numeric and string inputs
    
    Args:
        series: Pandas series to map
        mapping: Dictionary of mapping values
    
    Returns:
        Mapped pandas series with categories
    """
    try:
        # Check if series already contains string values
        if series.dtype == 'object' or series.dtype == 'string':
            return series.map(lambda x: mapping.get(x, 'Unknown'))
        
        # For numeric values, proceed with rounding
        rounded = series.fillna(-1).round()
        integers = rounded.astype(int)
        return integers.map(lambda x: mapping.get(x, 'Unknown'))
    except Exception as e:
        print(f"Error in safe_map: {e}")
        return pd.Series(['Unknown'] * len(series))

# 3. Main processing function
def process_maintenance_metrics(df):
    """
    Process maintenance metric columns including both categorical mappings 
    and scaled value groupings.
    """
    try:
        df_processed = df.copy()
        
        # Define scaled value mappings
        scaled_mappings = {
            'Service_History': {
                'ranges': [-1.0, -0.5, 0.5, 1.0],
                'labels': ['Poor Service', 'Basic Service', 'Regular Service', 'Excellent Service'],
                'colors': ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
            },
            'Fuel_Efficiency': {
                'ranges': [-1.0, -0.3, 0.3, 1.0],
                'labels': ['Low Efficiency', 'Moderate Efficiency', 'Good Efficiency', 'Excellent Efficiency'],
                'colors': ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
            },
            'Reported_Issues': {
                'ranges': [-1.0, -0.5, 0.5, 1.0],
                'labels': ['Few Issues', 'Some Issues', 'Multiple Issues', 'Many Issues'],
                'colors': ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            },
            'Vehicle_Age': {
                'ranges': [-1.0, -0.3, 0.3, 1.0],
                'labels': ['New', 'Relatively New', 'Mature', 'Older'],
                'colors': ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            },
            'Odometer_Reading': {
                'ranges': [-1.0, -0.3, 0.3, 1.0],
                'labels': ['Low Mileage', 'Average Mileage', 'High Mileage', 'Very High Mileage'],
                'colors': ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            }
        }
        
        # Process scaled numeric columns
        for column, mapping in scaled_mappings.items():
            if column in df_processed.columns:
                df_processed[f'{column}_CAT'] = df_processed[column].apply(
                    lambda x: group_scaled_values(x, mapping['ranges'], mapping['labels'])
                )
        
        # Define category mappings
        maintenance_history_map = {
            0: 'Poor',
            1: 'Average',
            2: 'Good',
            'Poor': 'Poor',      # Handle if already string
            'Average': 'Average',
            'Good': 'Good'
        }
        
        condition_map = {
            0: 'Worn out',
            1: 'Good',
            2: 'New',
            'Worn out': 'Worn out',  # Handle if already string
            'Good': 'Good',
            'New': 'New'
        }
        
        owner_type_map = {
            0: '1st Owner',
            1: '2nd Owner',
            2: '3rd Owner',
            '1st Owner': '1st Owner',  # Handle if already string
            '2nd Owner': '2nd Owner',
            '3rd Owner': '3rd Owner'
        }
        
        battery_map = {
            0: 'Weak',
            1: 'Good',
            2: 'New',
            'Weak': 'Weak',  # Handle if already string
            'Good': 'Good',
            'New': 'New'
        }
        
        # Apply mappings with error handling
        mapping_configs = [
            ('Maintenance_History_Code', 'Maintenance_History_CAT', maintenance_history_map),
            ('Tire_Condition_Code', 'Tire_Condition_CAT', condition_map),
            ('Brake_Condition_Code', 'Brake_Condition_CAT', condition_map),
            ('Battery_Status_Code', 'Battery_Status_CAT', battery_map),
            ('Owner_Type_Code', 'Owner_Type_CAT', owner_type_map)
        ]
        
        for source_col, target_col, mapping in mapping_configs:
            if source_col in df_processed.columns:
                df_processed[target_col] = safe_map(df_processed[source_col], mapping)
        
        # Create color schemes
        condition_colors = {
            'Poor': '#e74c3c',
            'Average': '#f39c12',
            'Good': '#2ecc71',
            'Worn out': '#e74c3c',
            'New': '#2ecc71',
            'Weak': '#e74c3c',
            'Unknown': '#95a5a6'
        }
        
        # Add colors from scaled mappings
        for mapping in scaled_mappings.values():
            for label, color in zip(mapping['labels'], mapping['colors']):
                condition_colors[label] = color
        
        # Create maintenance categories descriptions
        maintenance_categories = {
            'Maintenance_History': {
                'Poor': 'Irregular or minimal maintenance record',
                'Average': 'Regular but basic maintenance',
                'Good': 'Comprehensive and timely maintenance',
                'Unknown': 'Maintenance history not available'
            },
            'Component_Condition': {
                'Worn out': 'Requires immediate replacement/service',
                'Good': 'Functional with acceptable wear',
                'New': 'Recently replaced or minimal wear',
                'Unknown': 'Condition not assessed'
            }
        }
        
        return df_processed, condition_colors, maintenance_categories, scaled_mappings
        
    except Exception as e:
        print(f"Error in process_maintenance_metrics: {e}")
        raise

# 4. Insights generation function
def add_maintenance_insights(df):
    """
    Calculate key maintenance insights from the data with error handling
    """
    try:
        insights = {
            'maintenance_quality': {
                'good': df['Maintenance_History_CAT'].value_counts().get('Good', 0) / len(df) * 100,
                'average': df['Maintenance_History_CAT'].value_counts().get('Average', 0) / len(df) * 100,
                'poor': df['Maintenance_History_CAT'].value_counts().get('Poor', 0) / len(df) * 100
            },
            'component_health': {
                'tires_need_replacement': df['Tire_Condition_CAT'].value_counts().get('Worn out', 0) / len(df) * 100,
                'brakes_need_service': df['Brake_Condition_CAT'].value_counts().get('Worn out', 0) / len(df) * 100,
                'battery_weak': df['Battery_Status_CAT'].value_counts().get('Weak', 0) / len(df) * 100
            }
        }
        
        return insights
        
    except Exception as e:
        print(f"Error in add_maintenance_insights: {e}")
        return {
            'maintenance_quality': {'good': 0, 'average': 0, 'poor': 0},
            'component_health': {'tires_need_replacement': 0, 'brakes_need_service': 0, 'battery_weak': 0}
        }



# Placeholder functions for additional pages
def show_eda():
    # Clean the dataset first
    balanced_data_cleaned = clean_categorical_columns(balanced_data)

    st.title("Exploratory Data Analysis")
    
    # Create main tabs for different analysis types
    tab1, tab2, tab3 = st.tabs([
        "Univariate Analysis", 
        "Bivariate Analysis",
        "Multivariate Analysis"
    ])

    # Univariate Analysis Tab
    with tab1:
        st.header("Univariate Analysis")
        
        # Create sub-tabs for different variable groups
        uni_tab1, uni_tab2, uni_tab3, uni_tab4 = st.tabs([
            "Customer Demographics",
            "Vehicle Characteristics",
            "Risk Indicators",
            "Maintenance Metrics"
        ])
        
        # Customer Demographics Tab
        with uni_tab1:
            st.subheader("Customer Demographics Analysis")
            
            # Define demographic columns
            demographic_numeric = ['AGE', 'INCOME', 'HOME_VAL', 'YOJ']
            demographic_categorical = ['EDUCATION', 'OCCUPATION', 'GENDER', 'MSTATUS', 'PARENT1']
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### Numeric Variables Analysis")
                # Variable selector for numeric variables
                selected_numeric = st.selectbox(
                    "Select Numeric Variable",
                    demographic_numeric
                )
                
                # Plot type selector
                plot_type = st.radio(
                    "Select Plot Type",
                    ["Histogram", "Box Plot", "Violin Plot"]
                )
                
                # Create figure based on selection
                fig = go.Figure()
                
                if plot_type == "Histogram":
                    # Add histogram with KDE
                    fig.add_trace(go.Histogram(
                        x=balanced_data_cleaned[selected_numeric],
                        name="Distribution",
                        nbinsx=st.slider("Number of Bins", 10, 100, 50),
                        histnorm='probability density'
                    ))
                    
                    # Add KDE
                    kde = gaussian_kde(balanced_data_cleaned[selected_numeric].dropna())
                    x_range = np.linspace(
                        balanced_data_cleaned[selected_numeric].min(),
                        balanced_data_cleaned[selected_numeric].max(),
                        100
                    )
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        name="KDE",
                        line=dict(color='red')
                    ))
                
                elif plot_type == "Box Plot":
                    fig.add_trace(go.Box(
                        y=balanced_data_cleaned[selected_numeric],
                        name=selected_numeric,
                        boxpoints='outliers'
                    ))
                
                else:  # Violin Plot
                    fig.add_trace(go.Violin(
                        y=balanced_data_cleaned[selected_numeric],
                        name=selected_numeric,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_numeric} Distribution",
                    xaxis_title=selected_numeric,
                    yaxis_title="Frequency" if plot_type == "Histogram" else "Value",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.write("#### Summary Statistics")
                summary_stats = balanced_data_cleaned[selected_numeric].describe()
                st.dataframe(summary_stats)
            
            with col2:
                st.write("#### Categorical Variables Analysis")
                # Variable selector for categorical variables
                selected_categorical = st.selectbox(
                    "Select Categorical Variable",
                    demographic_categorical
                )
                
                # Plot type selector for categorical
                cat_plot_type = st.radio(
                    "Select Plot Type",
                    ["Bar Plot", "Pie Chart"],
                    key="cat_plot_type"
                )
                
                # Calculate value counts
                value_counts = balanced_data_cleaned[selected_categorical].value_counts()
                
                # Create figure based on selection
                if cat_plot_type == "Bar Plot":
                    fig = go.Figure(data=[
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            text=value_counts.values,
                            textposition='auto',
                        )
                    ])
                else:  # Pie Chart
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=value_counts.index,
                            values=value_counts.values,
                            hole=0.3,
                            textinfo='percent+label'
                        )
                    ])
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_categorical} Distribution",
                    height=400,
                    showlegend=True if cat_plot_type == "Pie Chart" else False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display frequency table
                st.write("#### Frequency Table")
                freq_df = pd.DataFrame({
                    'Count': value_counts,
                    'Percentage': (value_counts / len(balanced_data_cleaned) * 100).round(2)
                })
                st.dataframe(freq_df)
        
        # Other tabs remain as placeholders for now
        with uni_tab2:  
            st.subheader("Vehicle Characteristics Analysis")
            
            # Define vehicle-related columns
            vehicle_numeric = [
                'max_torque', 'max_power', 'displacement', 'turning_radius',
                'length', 'width', 'height', 'gross_weight', 'Mileage',
                'Engine_Size', 'CAR_AGE'
            ]
            
            vehicle_categorical = [
                'CAR_TYPE_Panel Truck', 'CAR_TYPE_Pickup', 'CAR_TYPE_SUV',
                'CAR_TYPE_Sports Car', 'CAR_TYPE_Van', 'segment_B1', 'segment_B2',
                'segment_C1', 'segment_C2', 'segment_Utility', 'fuel_type_Diesel',
                'fuel_type_Petrol'
            ]
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### Performance & Dimensions Analysis")
                # Variable selector for numeric variables
                selected_numeric = st.selectbox(
                    "Select Vehicle Metric",
                    vehicle_numeric,
                    key="vehicle_numeric"
                )
                
                # Plot type selector
                plot_type = st.radio(
                    "Select Plot Type",
                    ["Histogram", "Box Plot", "Violin Plot"],
                    key="vehicle_plot_type"
                )
                
                # Create figure based on selection
                fig = go.Figure()
                
                if plot_type == "Histogram":
                    # Add histogram with KDE
                    fig.add_trace(go.Histogram(
                        x=balanced_data_cleaned[selected_numeric],
                        name="Distribution",
                        nbinsx=st.slider("Number of Bins", 10, 100, 50, key="vehicle_bins"),
                        histnorm='probability density'
                    ))
                    
                    # Add KDE
                    kde = gaussian_kde(balanced_data_cleaned[selected_numeric].dropna())
                    x_range = np.linspace(
                        balanced_data_cleaned[selected_numeric].min(),
                        balanced_data_cleaned[selected_numeric].max(),
                        100
                    )
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        name="KDE",
                        line=dict(color='red')
                    ))
                
                elif plot_type == "Box Plot":
                    fig.add_trace(go.Box(
                        y=balanced_data_cleaned[selected_numeric],
                        name=selected_numeric,
                        boxpoints='outliers'
                    ))
                
                else:  # Violin Plot
                    fig.add_trace(go.Violin(
                        y=balanced_data_cleaned[selected_numeric],
                        name=selected_numeric,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_numeric} Distribution",
                    xaxis_title=selected_numeric,
                    yaxis_title="Frequency" if plot_type == "Histogram" else "Value",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.write("#### Summary Statistics")
                summary_stats = balanced_data_cleaned[selected_numeric].describe()
                st.dataframe(summary_stats)
            
            with col2:
                st.write("#### Vehicle Type & Features Analysis")
                
                # Combine car types into a single column for better visualization
                car_types = balanced_data_cleaned[[
                    'CAR_TYPE_Panel Truck', 'CAR_TYPE_Pickup', 'CAR_TYPE_SUV',
                    'CAR_TYPE_Sports Car', 'CAR_TYPE_Van'
                ]].idxmax(axis=1).map(lambda x: x.replace('CAR_TYPE_', ''))
                
                # Calculate car type distribution
                car_type_dist = car_types.value_counts()
                
                # Create pie chart for car types
                fig_car_types = go.Figure(data=[
                    go.Pie(
                        labels=car_type_dist.index,
                        values=car_type_dist.values,
                        hole=0.3,
                        textinfo='percent+label'
                    )
                ])
                
                fig_car_types.update_layout(
                    title="Distribution of Vehicle Types",
                    height=400
                )
                
                st.plotly_chart(fig_car_types, use_container_width=True)
                
                # Segment analysis
                st.write("#### Vehicle Segment Analysis")
                
                # Combine segments into a single column
                segments = balanced_data_cleaned[[
                    'segment_B1', 'segment_B2', 'segment_C1',
                    'segment_C2', 'segment_Utility'
                ]].idxmax(axis=1).map(lambda x: x.replace('segment_', ''))
                
                segment_dist = segments.value_counts()
                
                # Create bar chart for segments
                fig_segments = go.Figure(data=[
                    go.Bar(
                        x=segment_dist.index,
                        y=segment_dist.values,
                        text=segment_dist.values,
                        textposition='auto'
                    )
                ])
                
                fig_segments.update_layout(
                    title="Distribution of Vehicle Segments",
                    xaxis_title="Segment",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig_segments, use_container_width=True)
                
            # Additional section for fuel type analysis
            st.write("#### Fuel Type Analysis")
            
            # Calculate fuel type distribution
            fuel_types = balanced_data_cleaned[[
                'fuel_type_Diesel', 'fuel_type_Petrol'
            ]].idxmax(axis=1).map(lambda x: x.replace('fuel_type_', ''))
            
            fuel_dist = fuel_types.value_counts()
            
            # Create horizontal bar chart for fuel types
            fig_fuel = go.Figure(data=[
                go.Bar(
                    y=fuel_dist.index,
                    x=fuel_dist.values,
                    text=fuel_dist.values,
                    textposition='auto',
                    orientation='h'
                )
            ])
            
            fig_fuel.update_layout(
                title="Distribution of Fuel Types",
                xaxis_title="Count",
                yaxis_title="Fuel Type",
                height=300
            )
            
            st.plotly_chart(fig_fuel, use_container_width=True)
            
            # Add insights section
            st.write("#### Key Insights")
            st.markdown("""
            - Vehicle Type Distribution shows the relative market share of different vehicle categories
            - Segment Analysis reveals the positioning of vehicles in different market segments
            - Performance Metrics (power, torque, displacement) indicate the technical capabilities
            - Dimensional Analysis shows the physical characteristics distribution
            - Fuel Type Distribution indicates the prevalence of different fuel technologies
            """)




        with uni_tab3:
            st.subheader("Risk Indicators Analysis")
            
            # Process the data with our new mappings
            processed_data, accident_colors = process_risk_indicators(balanced_data_cleaned)
            
            # Define risk-related columns
            risk_numeric = [
                'CLM_FREQ',   # Claim frequency
                'MVR_PTS',    # Motor vehicle record points
                'CLM_AMT',    # Claim amount
                'OLDCLAIM',   # Past claim amount
                'TIF'         # Time in force (policy duration)
            ]
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### Claims and Violations Metrics")
                # Variable selector for numeric risk indicators
                selected_numeric = st.selectbox(
                    "Select Risk Metric",
                    risk_numeric,
                    format_func=lambda x: {
                        'CLM_FREQ': 'Claim Frequency',
                        'MVR_PTS': 'Motor Vehicle Points',
                        'CLM_AMT': 'Claim Amount',
                        'OLDCLAIM': 'Previous Claim Amount',
                        'TIF': 'Policy Duration'
                    }[x],
                    key="risk_numeric"
                )
                
                # Plot type selector
                plot_type = st.radio(
                    "Select Plot Type",
                    ["Histogram", "Box Plot"],
                    key="risk_plot_type"
                )
                
                # Create figure based on selection
                fig = go.Figure()
                
                if plot_type == "Histogram":
                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=processed_data[selected_numeric],
                        name="Distribution",
                        nbinsx=st.slider("Number of Bins", 10, 100, 50, key="risk_bins")
                    ))
                    
                else:  # Box Plot
                    fig.add_trace(go.Box(
                        y=processed_data[selected_numeric],
                        name=selected_numeric,
                        boxpoints='outliers'
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_numeric} Distribution",
                    xaxis_title=selected_numeric,
                    yaxis_title="Frequency" if plot_type == "Histogram" else "Value",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.write("#### Summary Statistics")
                summary_stats = processed_data[selected_numeric].describe().round(2)
                st.dataframe(summary_stats)
            
            with col2:
                st.write("#### Risk Categories Distribution")
                
                # Create tabs for different categorical analyses
                cat_tab1, cat_tab2 = st.tabs(["License & Usage", "Location & Accidents"])
                
                with cat_tab1:
                    # License Revocation Status with updated categories
                    revoked_dist = processed_data['REVOKED_CAT'].value_counts()
                    fig_revoked = go.Figure(data=[
                        go.Pie(
                            labels=revoked_dist.index,
                            values=revoked_dist.values,
                            hole=0.3,
                            textinfo='percent+label',
                            marker_colors=['#2ecc71', '#e74c3c']  # Green for Not Revoked, Red for Revoked
                        )
                    ])
                    fig_revoked.update_layout(
                        title="License Revocation Status",
                        height=300
                    )
                    st.plotly_chart(fig_revoked, use_container_width=True)
                    
                    # Car Usage Analysis with updated categories
                    car_use_dist = processed_data['CAR_USE_CAT'].value_counts()
                    fig_use = go.Figure(data=[
                        go.Bar(
                            x=car_use_dist.index,
                            y=car_use_dist.values,
                            text=car_use_dist.values,
                            textposition='auto',
                            marker_color=['#3498db', '#f1c40f']  # Blue for Private, Yellow for Commercial
                        )
                    ])
                    fig_use.update_layout(
                        title="Vehicle Usage Distribution",
                        xaxis_title="Usage Type",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig_use, use_container_width=True)
                
                with cat_tab2:
                    # Urbanicity Analysis with updated categories
                    urban_dist = processed_data['URBANICITY_CAT'].value_counts()
                    fig_urban = go.Figure(data=[
                        go.Pie(
                            labels=urban_dist.index,
                            values=urban_dist.values,
                            hole=0.3,
                            textinfo='percent+label',
                            marker_colors=['#95a5a6', '#34495e']  # Light gray for Low, Dark gray for High
                        )
                    ])
                    fig_urban.update_layout(
                        title="Urbanicity Distribution",
                        height=300
                    )
                    st.plotly_chart(fig_urban, use_container_width=True)
                    
                    # Accident History Analysis with updated categories
                    accident_dist = processed_data['ACCIDENT_HISTORY_CAT'].value_counts()
                    colors = [accident_colors[cat] for cat in accident_dist.index]
                    fig_accident = go.Figure(data=[
                        go.Bar(
                            x=accident_dist.index,
                            y=accident_dist.values,
                            text=accident_dist.values,
                            textposition='auto',
                            marker_color=colors
                        )
                    ])
                    fig_accident.update_layout(
                        title="Accident History Distribution",
                        xaxis_title="Accident History Category",
                        yaxis_title="Count",
                        height=300,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_accident, use_container_width=True)
            
            # Risk Category Descriptions
            st.write("#### Risk Category Descriptions")
            descriptions = get_risk_category_descriptions()
            
            # Display descriptions in an expandable section
            with st.expander("Click to view category descriptions"):
                for category, desc in descriptions.items():
                    st.write(f"**{category}**")
                    for subcategory, explanation in desc.items():
                        st.write(f"- {subcategory}: {explanation}")
            
            # Key Risk Insights based on actual data
            st.write("#### Key Risk Insights")
            
            # Calculate some key metrics for insights
            claim_freq_mean = processed_data['CLM_FREQ'].mean()
            high_risk_drivers = (processed_data['MVR_PTS'] > 6).sum() / len(processed_data) * 100
            commercial_pct = (processed_data['CAR_USE'] == 1).sum() / len(processed_data) * 100
            
            st.markdown(f"""
            - **Claims Pattern**: Average claim frequency is {claim_freq_mean:.2f} claims per policy
            - **High-Risk Drivers**: {high_risk_drivers:.1f}% of drivers have significant motor vehicle points (>6)
            - **Vehicle Usage**: {commercial_pct:.1f}% of vehicles are used for commercial purposes
            - **Geographic Distribution**: See urbanicity chart for urban/rural split
            - **Accident Severity**: Distribution shown in accident history chart above
            """)
            
        with uni_tab4:
            st.subheader("Maintenance Metrics Analysis")
            
            # Process maintenance data with our mappings - now capturing all 4 return values
            processed_data, condition_colors, maintenance_categories, scaled_mappings = process_maintenance_metrics(balanced_data_cleaned)
            
            # Define maintenance-related columns
            maintenance_numeric = [
                'Service_History',      
                'Fuel_Efficiency',      
                'Reported_Issues',      
                'Vehicle_Age',          
                'Odometer_Reading'      
            ]
            
            maintenance_categorical = [
                'Maintenance_History_CAT',  
                'Tire_Condition_CAT',       
                'Brake_Condition_CAT',      
                'Battery_Status_CAT',
                'Owner_Type_CAT'            
            ]
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### Vehicle Service Metrics")
                
                # Variable selector for numeric maintenance indicators
                selected_numeric = st.selectbox(
                    "Select Service Metric",
                    maintenance_numeric,
                    key="maintenance_numeric"
                )
                
                # Using scaled mappings for better labels
                display_name = scaled_mappings.get(selected_numeric, {}).get('labels', [selected_numeric])[0]
                st.write(f"Analyzing: {display_name}")
                
                # Plot type selector
                plot_type = st.radio(
                    "Select Plot Type",
                    ["Distribution Plot", "Box Plot"],
                    key="maintenance_plot_type"
                )
                
                fig = go.Figure()
                
                if plot_type == "Distribution Plot":
                    fig.add_trace(go.Histogram(
                        x=processed_data[selected_numeric],
                        nbinsx=30,
                        name="Distribution"
                    ))
                    
                    # Add category ranges if available
                    if selected_numeric in scaled_mappings:
                        for i, (range_val, label) in enumerate(zip(
                            scaled_mappings[selected_numeric]['ranges'],
                            scaled_mappings[selected_numeric]['labels']
                        )):
                            fig.add_vline(
                                x=range_val,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=label
                            )
                    
                else:  # Box Plot
                    fig.add_trace(go.Box(
                        y=processed_data[selected_numeric],
                        name=display_name,
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title=f"{display_name} Analysis",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add metric summary
                metric_stats = processed_data[selected_numeric].describe()
                st.write("#### Summary Statistics")
                st.dataframe(metric_stats.round(2))
            
            with col2:
                st.write("#### Component Conditions")
                
                # Create tabs for conditions and history
                cond_tab1, cond_tab2 = st.tabs(["Critical Components", "Maintenance & Ownership"])
                
                with cond_tab1:
                    # Tire Condition Analysis
                    tire_dist = processed_data['Tire_Condition_CAT'].value_counts()
                    fig_tire = go.Figure(data=[
                        go.Bar(
                            x=tire_dist.index,
                            y=tire_dist.values,
                            text=tire_dist.values,
                            textposition='auto',
                            marker_color=[condition_colors[cat] for cat in tire_dist.index]
                        )
                    ])
                    fig_tire.update_layout(
                        title="Tire Condition Status",
                        height=300
                    )
                    st.plotly_chart(fig_tire, use_container_width=True)
                    
                    # Brake Condition Analysis
                    brake_dist = processed_data['Brake_Condition_CAT'].value_counts()
                    fig_brake = go.Figure(data=[
                        go.Bar(
                            x=brake_dist.index,
                            y=brake_dist.values,
                            text=brake_dist.values,
                            textposition='auto',
                            marker_color=[condition_colors[cat] for cat in brake_dist.index]
                        )
                    ])
                    fig_brake.update_layout(
                        title="Brake System Status",
                        height=300
                    )
                    st.plotly_chart(fig_brake, use_container_width=True)
                
                with cond_tab2:
                    # Maintenance History Analysis
                    maint_dist = processed_data['Maintenance_History_CAT'].value_counts()
                    fig_maint = go.Figure(data=[
                        go.Pie(
                            labels=maint_dist.index,
                            values=maint_dist.values,
                            hole=0.3,
                            marker_colors=[condition_colors[cat] for cat in maint_dist.index]
                        )
                    ])
                    fig_maint.update_layout(
                        title="Overall Maintenance History",
                        height=300
                    )
                    st.plotly_chart(fig_maint, use_container_width=True)
                    
                    # Owner Type Analysis
                    owner_dist = processed_data['Owner_Type_CAT'].value_counts()
                    fig_owner = go.Figure(data=[
                        go.Bar(
                            x=owner_dist.index,
                            y=owner_dist.values,
                            text=owner_dist.values,
                            textposition='auto',
                            marker_color=['#3498db', '#9b59b6', '#95a5a6']
                        )
                    ])
                    fig_owner.update_layout(
                        title="Vehicle Ownership Distribution",
                        height=300
                    )
                    st.plotly_chart(fig_owner, use_container_width=True)
            
            # Calculate maintenance insights
            insights = add_maintenance_insights(processed_data)
            
            # Overall Maintenance Summary
            st.write("#### Maintenance Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Vehicles Needing Service",
                    f"{insights['maintenance_quality']['poor']:.1f}%",
                    delta=None
                )
            with col2:
                st.metric(
                    "Well-Maintained Vehicles",
                    f"{insights['maintenance_quality']['good']:.1f}%",
                    delta=None
                )
            with col3:
                st.metric(
                    "Components Needing Attention",
                    f"{max(insights['component_health'].values()):.1f}%",
                    delta=None
                )
            
            # Detailed insights in expander
            with st.expander("View Detailed Maintenance Insights"):
                st.markdown("""
                #### Component Health Indicators
                """)
                
                for component, value in insights['component_health'].items():
                    st.write(f"- {component.replace('_', ' ').title()}: {value:.1f}%")
                
                st.markdown("""
                #### Maintenance Categories
                """)
                
                for category, descriptions in maintenance_categories.items():
                    st.write(f"**{category.replace('_', ' ')}**")
                    for status, desc in descriptions.items():
                        st.write(f"- {status}: {desc}")

    # Bivariate Analysis Tab
# Bivariate Analysis Tab
    with tab2:
        st.header("Bivariate Analysis")
        
        # Create sub-tabs for different target variables
        bi_tab1, bi_tab2, bi_tab3, bi_tab4 = st.tabs([
            "Claims Analysis (CLAIM_FLAG)",
            "Maintenance Analysis (Need_Maintenance)",
            "Insurance Claims (is_claim)",
            "Cross-Target Analysis"
        ])
        
        # Claims Analysis Tab
        with bi_tab1:
            st.subheader("Claims Analysis")
            
            # Create sections for different analysis types
            sections = st.tabs([
                "Demographic Relationships",
                "Vehicle Characteristics",
                "Safety Features"
            ])
            
            # 1. Demographic Relationships
            with sections[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age vs Claims (Box Plot)
                    fig_age = go.Figure()
                    
                    # Create box plot for each claim category
                    for claim in [0, 1]:
                        age_data = balanced_data_cleaned[balanced_data_cleaned['CLAIM_FLAG'] == claim]['AGE']
                        fig_age.add_trace(go.Box(
                            y=age_data,
                            name=f"{'Claim' if claim == 1 else 'No Claim'}",
                            boxpoints='outliers'
                        ))
                    
                    fig_age.update_layout(
                        title="Age Distribution by Claim Status",
                        yaxis_title="Age",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_age, use_container_width=True)
                    
                    # Education vs Claims (Stacked Bar Chart)
                    edu_claim = pd.crosstab(
                        balanced_data_cleaned['EDUCATION'],
                        balanced_data_cleaned['CLAIM_FLAG'],
                        normalize='index'
                    ) * 100
                    
                    fig_edu = go.Figure(data=[
                        go.Bar(name='No Claim', x=edu_claim.index, y=edu_claim[0]),
                        go.Bar(name='Claim', x=edu_claim.index, y=edu_claim[1])
                    ])
                    
                    fig_edu.update_layout(
                        barmode='stack',
                        title="Claims Distribution by Education Level",
                        xaxis_title="Education Level",
                        yaxis_title="Percentage",
                        height=400
                    )
                    
                    st.plotly_chart(fig_edu, use_container_width=True)
                
                with col2:
                    # Income vs Claims (Violin Plot)
                    fig_income = go.Figure()
                    
                    for claim in [0, 1]:
                        income_data = balanced_data_cleaned[balanced_data_cleaned['CLAIM_FLAG'] == claim]['INCOME']
                        fig_income.add_trace(go.Violin(
                            y=income_data,
                            name=f"{'Claim' if claim == 1 else 'No Claim'}",
                            box_visible=True,
                            meanline_visible=True
                        ))
                    
                    fig_income.update_layout(
                        title="Income Distribution by Claim Status",
                        yaxis_title="Income",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_income, use_container_width=True)
                    
                    # Occupation vs Claims (Heatmap)
                    occ_claim = pd.crosstab(
                        balanced_data_cleaned['OCCUPATION'],
                        balanced_data_cleaned['CLAIM_FLAG'],
                        normalize='index'
                    ) * 100
                    
                    fig_occ = go.Figure(data=go.Heatmap(
                        z=occ_claim.values,
                        x=['No Claim', 'Claim'],
                        y=occ_claim.index,
                        colorscale='RdYlBu',
                        text=np.round(occ_claim.values, 1),
                        texttemplate='%{text}%',
                        textfont={"size": 10},
                        showscale=True
                    ))
                    
                    fig_occ.update_layout(
                        title="Claims Percentage by Occupation",
                        height=400
                    )
                    
                    st.plotly_chart(fig_occ, use_container_width=True)
            
            # 2. Vehicle Characteristics
            with sections[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Car Age vs Claims (Box Plot)
                    fig_car_age = go.Figure()
                    
                    for claim in [0, 1]:
                        car_age_data = balanced_data_cleaned[balanced_data_cleaned['CLAIM_FLAG'] == claim]['CAR_AGE']
                        fig_car_age.add_trace(go.Box(
                            y=car_age_data,
                            name=f"{'Claim' if claim == 1 else 'No Claim'}",
                            boxpoints='outliers'
                        ))
                    
                    fig_car_age.update_layout(
                        title="Car Age Distribution by Claim Status",
                        yaxis_title="Car Age",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_car_age, use_container_width=True)
                
                with col2:
                    # Vehicle Type vs Claims (Stacked Bar)
                    car_types = ['CAR_TYPE_Panel Truck', 'CAR_TYPE_Pickup', 
                               'CAR_TYPE_SUV', 'CAR_TYPE_Sports Car', 'CAR_TYPE_Van']
                    
                    # Get the most common car type for each row
                    balanced_data_cleaned['CAR_TYPE'] = balanced_data_cleaned[car_types].idxmax(axis=1).str.replace('CAR_TYPE_', '')
                    
                    type_claim = pd.crosstab(
                        balanced_data_cleaned['CAR_TYPE'],
                        balanced_data_cleaned['CLAIM_FLAG'],
                        normalize='index'
                    ) * 100
                    
                    fig_type = go.Figure(data=[
                        go.Bar(name='No Claim', x=type_claim.index, y=type_claim[0]),
                        go.Bar(name='Claim', x=type_claim.index, y=type_claim[1])
                    ])
                    
                    fig_type.update_layout(
                        barmode='stack',
                        title="Claims Distribution by Vehicle Type",
                        xaxis_title="Vehicle Type",
                        yaxis_title="Percentage",
                        height=400
                    )
                    
                    st.plotly_chart(fig_type, use_container_width=True)
            
            # 3. Safety Features
            with sections[2]:
                # Create a list of safety feature columns with their display names
                safety_features = {
                    'airbags': 'Airbags Present',
                    'is_esc': 'Electronic Stability Control',
                    'is_adjustable_steering': 'Adjustable Steering',
                    'is_tpms': 'Tire Pressure Monitoring',
                    'is_parking_sensors': 'Parking Sensors',
                    'is_parking_camera': 'Parking Camera'
                }
                
                # Calculate claim rates for each safety feature
                claim_rates_data = []
                
                for feature, display_name in safety_features.items():
                    # Calculate crosstab with absolute numbers
                    cross_tab = pd.crosstab(
                        balanced_data_cleaned[feature],
                        balanced_data_cleaned['CLAIM_FLAG']
                    )
                    
                    # Calculate percentages
                    percentages = (cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100).round(1)
                    
                    # Store the claim rate (percentage of claims) for each feature value
                    if 1 in percentages.columns:  # If there are claims
                        claim_rates_data.append({
                            'Feature': display_name,
                            'No_Feature_Claim_Rate': percentages.loc[0, 1] if 0 in percentages.index else 0,
                            'With_Feature_Claim_Rate': percentages.loc[1, 1] if 1 in percentages.index else 0
                        })
                
                # Convert to DataFrame
                claim_rates_df = pd.DataFrame(claim_rates_data)
                
                # Create heatmap
                fig_safety = go.Figure()
                
                # Add heatmap trace
                fig_safety.add_trace(go.Heatmap(
                    z=np.array([claim_rates_df['No_Feature_Claim_Rate'], 
                               claim_rates_df['With_Feature_Claim_Rate']]).T,
                    x=['Without Feature', 'With Feature'],
                    y=claim_rates_df['Feature'],
                    colorscale='RdYlBu_r',  # Reversed scale: Red (high claims) to Blue (low claims)
                    text=np.array([claim_rates_df['No_Feature_Claim_Rate'], 
                                 claim_rates_df['With_Feature_Claim_Rate']]).T,
                    texttemplate='%{text:.1f}%',
                    textfont={"size": 12},
                    showscale=True,
                    colorbar=dict(title='Claim Rate (%)')
                ))
                
                # Update layout
                fig_safety.update_layout(
                    title="Claim Rates by Safety Feature",
                    xaxis_title="Feature Status",
                    yaxis_title="Safety Feature",
                    height=500,
                    yaxis={'autorange': 'reversed'}  # To match the original order
                )
                
                st.plotly_chart(fig_safety, use_container_width=True)
                
                # Add feature impact analysis
                st.write("#### Safety Feature Impact Analysis")
                
                # Calculate and display the impact of each feature
                impact_data = []
                for idx, row in claim_rates_df.iterrows():
                    impact = row['No_Feature_Claim_Rate'] - row['With_Feature_Claim_Rate']
                    impact_data.append({
                        'Feature': row['Feature'],
                        'Impact': impact,
                        'Effectiveness': 'Positive' if impact > 0 else 'Negative' if impact < 0 else 'Neutral'
                    })
                
                impact_df = pd.DataFrame(impact_data)
                impact_df = impact_df.sort_values('Impact', ascending=False)
                
                # Display impact analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Feature Effectiveness (% Reduction in Claims):")
                    for _, row in impact_df.iterrows():
                        if row['Impact'] != 0:
                            color = "green" if row['Impact'] > 0 else "red"
                            st.markdown(f"- **{row['Feature']}**: "
                                      f"<span style='color:{color}'>{abs(row['Impact']):.1f}% "
                                      f"{'reduction' if row['Impact'] > 0 else 'increase'}</span>",
                                      unsafe_allow_html=True)
                
                with col2:
                    st.write("Key Insights:")
                    st.markdown("""
                    - Features showing claim reduction are more effective
                    - Red indicates potential correlation with higher claim rates
                    - Consider combinations of features for optimal safety
                    """)
                # Create a list of safety feature columns
                safety_features = [
                    'airbags', 'is_esc', 'is_adjustable_steering',
                    'is_tpms', 'is_parking_sensors', 'is_parking_camera'
                ]
                
                # Calculate claim rates for each safety feature
                safety_claims = pd.DataFrame()
                
                for feature in safety_features:
                    claim_rates = pd.crosstab(
                        balanced_data_cleaned[feature],
                        balanced_data_cleaned['CLAIM_FLAG'],
                        normalize='index'
                    ) * 100
                    safety_claims[feature] = claim_rates[1]  # Get claim rate (1)
                
                # Create heatmap for safety features vs claims
                fig_safety = go.Figure(data=go.Heatmap(
                    z=safety_claims.T.values,
                    x=['No', 'Yes'],
                    y=[feat.replace('is_', '').replace('_', ' ').title() for feat in safety_features],
                    colorscale='RdYlBu',
                    text=np.round(safety_claims.T.values, 1),
                    texttemplate='%{text}%',
                    textfont={"size": 10},
                    showscale=True
                ))
                
                fig_safety.update_layout(
                    title="Claim Rates by Safety Feature",
                    xaxis_title="Feature Present",
                    height=500
                )
                
                st.plotly_chart(fig_safety, use_container_width=True)
                
                # Add insights
                st.write("#### Key Insights from Claims Analysis")
                st.markdown("""
                1. **Demographic Insights:**
                   - Age distribution patterns between claim and no-claim groups
                   - Income levels' relationship with claim likelihood
                   - Educational and occupational patterns in claims
                
                2. **Vehicle Insights:**
                   - Relationship between car age and claim probability
                   - Vehicle type preferences and their claim rates
                
                3. **Safety Feature Impact:**
                   - Effectiveness of different safety features
                   - Correlation between safety features and claim rates
                """)

# Maintenance Analysis Tab (Need_Maintenance)
        with bi_tab2:
            st.subheader("Maintenance Analysis")
            
            # Vehicle Health Analysis
            st.write("#### Vehicle Health Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mileage vs Maintenance
                fig_mileage = go.Figure()
                
                for maint in [0, 1]:
                    mileage_data = balanced_data_cleaned[balanced_data_cleaned['Need_Maintenance'] == maint]['Odometer_Reading']
                    fig_mileage.add_trace(go.Box(
                        y=mileage_data,
                        name=f"{'Needs Maintenance' if maint == 1 else 'No Maintenance Needed'}",
                        boxpoints='outliers'
                    ))
                
                fig_mileage.update_layout(
                    title="Mileage Distribution by Maintenance Need",
                    yaxis_title="Odometer Reading",
                    height=400
                )
                st.plotly_chart(fig_mileage, use_container_width=True)
                
                # Service History vs Maintenance
                fig_service = go.Figure()
                
                for maint in [0, 1]:
                    service_data = balanced_data_cleaned[balanced_data_cleaned['Need_Maintenance'] == maint]['Service_History']
                    fig_service.add_trace(go.Violin(
                        y=service_data,
                        name=f"{'Needs Maintenance' if maint == 1 else 'No Maintenance Needed'}",
                        box_visible=True
                    ))
                
                fig_service.update_layout(
                    title="Service History by Maintenance Need",
                    yaxis_title="Service History Score",
                    height=400
                )
                st.plotly_chart(fig_service, use_container_width=True)
            
            with col2:
                # Vehicle Age vs Maintenance
                fig_age = go.Figure()
                
                age_maint = pd.crosstab(
                    pd.qcut(balanced_data_cleaned['Vehicle_Age'], q=5),
                    balanced_data_cleaned['Need_Maintenance'],
                    normalize='index'
                ) * 100
                
                fig_age.add_trace(go.Bar(
                    x=age_maint.index.astype(str),
                    y=age_maint[1],
                    name='Needs Maintenance',
                    marker_color='#e74c3c'
                ))
                
                fig_age.update_layout(
                    title="Maintenance Need by Vehicle Age Quintiles",
                    xaxis_title="Vehicle Age Groups",
                    yaxis_title="Percentage Needing Maintenance",
                    height=400
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            # Component Status Analysis
            st.write("#### Component Status Analysis")
            
            # Prepare component status data
            components = {
                'Battery_Status_Code': 'Battery Status',
                'Tire_Condition_Code': 'Tire Condition',
                'Brake_Condition_Code': 'Brake Condition'
            }
            
            # Create a combined heatmap for all components
            component_data = []
            
            for code, name in components.items():
                cross_tab = pd.crosstab(
                    balanced_data_cleaned[code],
                    balanced_data_cleaned['Need_Maintenance'],
                    normalize='index'
                ) * 100
                
                for condition in cross_tab.index:
                    component_data.append({
                        'Component': name,
                        'Condition': condition,
                        'Maintenance_Rate': cross_tab.loc[condition, 1]
                    })
            
            component_df = pd.DataFrame(component_data)
            
            fig_components = go.Figure(data=go.Heatmap(
                z=component_df['Maintenance_Rate'].values.reshape(3, 3),
                x=['Poor', 'Fair', 'Good'],
                y=list(components.values()),
                colorscale='RdYlBu_r',
                text=np.round(component_df['Maintenance_Rate'].values.reshape(3, 3), 1),
                texttemplate='%{text}%',
                textfont={"size": 12},
                colorbar=dict(title='Maintenance Need (%)')
            ))
            
            fig_components.update_layout(
                title="Maintenance Need by Component Condition",
                xaxis_title="Component Condition",
                yaxis_title="Component Type",
                height=400
            )
            
            st.plotly_chart(fig_components, use_container_width=True)
            
            # Key Insights
            st.write("#### Key Maintenance Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Vehicle Health Patterns:**
                - Relationship between mileage and maintenance needs
                - Impact of service history on maintenance requirements
                - Age-related maintenance patterns
                """)
            
            with col2:
                st.markdown("""
                **Component Status Impact:**
                - Critical component condition relationships
                - Maintenance need variations by component state
                - Component deterioration patterns
                """)
            
# Insurance Claims Analysis Tab (is_claim)
        with bi_tab3:
            st.subheader("Insurance Claims Analysis")
            
            # Risk Factors Analysis
            st.write("#### Risk Factor Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # MVR_PTS vs Claims
                fig_mvr = go.Figure()
                
                for claim in [0, 1]:
                    mvr_data = balanced_data_cleaned[balanced_data_cleaned['is_claim'] == claim]['MVR_PTS']
                    fig_mvr.add_trace(go.Box(
                        y=mvr_data,
                        name=f"{'Claim' if claim == 1 else 'No Claim'}",
                        boxpoints='outliers'
                    ))
                
                fig_mvr.update_layout(
                    title="Motor Vehicle Record Points Distribution by Claim Status",
                    yaxis_title="MVR Points",
                    height=400
                )
                st.plotly_chart(fig_mvr, use_container_width=True)
                
                # Previous Claims vs New Claims (OLDCLAIM)
                fig_prev = go.Figure()
                
                for claim in [0, 1]:
                    prev_claim_data = balanced_data_cleaned[balanced_data_cleaned['is_claim'] == claim]['OLDCLAIM']
                    fig_prev.add_trace(go.Violin(
                        y=prev_claim_data,
                        name=f"{'Current Claim' if claim == 1 else 'No Current Claim'}",
                        box_visible=True
                    ))
                
                fig_prev.update_layout(
                    title="Previous Claim Amount by Current Claim Status",
                    yaxis_title="Previous Claim Amount",
                    height=400
                )
                st.plotly_chart(fig_prev, use_container_width=True)
            
            with col2:
                # Urbanicity vs Claims
                urban_claims = pd.crosstab(
                    balanced_data_cleaned['URBANICITY'],
                    balanced_data_cleaned['is_claim'],
                    normalize='index'
                ) * 100
                
                fig_urban = go.Figure(data=[
                    go.Bar(
                        x=['Rural', 'Urban'],
                        y=urban_claims[1],
                        name='Claim Rate',
                        marker_color='#e74c3c'
                    )
                ])
                
                fig_urban.update_layout(
                    title="Claim Rate by Urbanicity",
                    xaxis_title="Area Type",
                    yaxis_title="Claim Rate (%)",
                    height=400
                )
                st.plotly_chart(fig_urban, use_container_width=True)
            
            # Safety Analysis
            st.write("#### Safety Analysis")
            
            # NCAP Rating vs Claims
            fig_ncap = go.Figure()
            
            ncap_claims = pd.crosstab(
                balanced_data_cleaned['ncap_rating'],
                balanced_data_cleaned['is_claim'],
                normalize='index'
            ) * 100
            
            fig_ncap.add_trace(go.Bar(
                x=ncap_claims.index,
                y=ncap_claims[1],
                name='Claim Rate',
                marker_color='#3498db'
            ))
            
            fig_ncap.update_layout(
                title="Claim Rate by NCAP Rating",
                xaxis_title="NCAP Rating",
                yaxis_title="Claim Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig_ncap, use_container_width=True)
            
            # Safety Features Analysis
            safety_features = [
                'airbags', 'is_esc', 'is_adjustable_steering',
                'is_tpms', 'is_parking_sensors', 'is_parking_camera'
            ]
            
            # Calculate claim rates for each safety feature
            safety_data = []
            
            for feature in safety_features:
                feature_claims = pd.crosstab(
                    balanced_data_cleaned[feature],
                    balanced_data_cleaned['is_claim'],
                    normalize='index'
                ) * 100
                
                if 1 in feature_claims.columns:
                    safety_data.append({
                        'Feature': feature.replace('is_', '').replace('_', ' ').title(),
                        'Without_Feature': feature_claims.loc[0, 1] if 0 in feature_claims.index else 0,
                        'With_Feature': feature_claims.loc[1, 1] if 1 in feature_claims.index else 0
                    })
            
            safety_df = pd.DataFrame(safety_data)
            
            # Create heatmap for safety features
            fig_safety = go.Figure(data=go.Heatmap(
                z=np.array([safety_df['Without_Feature'], safety_df['With_Feature']]).T,
                x=['Without Feature', 'With Feature'],
                y=safety_df['Feature'],
                colorscale='RdYlBu_r',
                text=np.array([safety_df['Without_Feature'], safety_df['With_Feature']]).T,
                texttemplate='%{text:.1f}%',
                textfont={"size": 12},
                colorbar=dict(title='Claim Rate (%)')
            ))
            
            fig_safety.update_layout(
                title="Claim Rates by Safety Feature",
                xaxis_title="Feature Status",
                yaxis_title="Safety Feature",
                height=500
            )
            
            st.plotly_chart(fig_safety, use_container_width=True)
            
            # Key Insights
            st.write("#### Key Risk and Safety Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Risk Factor Patterns:**
                - Relationship between MVR points and claim likelihood
                - Impact of previous claims on new claim probability
                - Urban vs rural claim rate differences
                """)
            
            with col2:
                st.markdown("""
                **Safety Impact:**
                - NCAP rating influence on claim rates
                - Effectiveness of different safety features
                - Combined safety feature implications
                """)

# Cross-Target Analysis Tab
        with bi_tab4:
            st.subheader("Cross-Target Analysis")
            
            # Maintenance vs Claims Analysis
            st.write("#### Maintenance and Claims Relationship")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Maintenance vs CLAIM_FLAG
                maint_claim = pd.crosstab(
                    balanced_data_cleaned['Need_Maintenance'],
                    balanced_data_cleaned['CLAIM_FLAG'],
                    normalize='index'
                ) * 100
                
                fig_maint_claim = go.Figure(data=[
                    go.Bar(
                        x=['No Maintenance Needed', 'Maintenance Needed'],
                        y=maint_claim[1],
                        name='Claim Rate',
                        marker_color='#e74c3c'
                    )
                ])
                
                fig_maint_claim.update_layout(
                    title="Claim Rate by Maintenance Status",
                    xaxis_title="Maintenance Status",
                    yaxis_title="Claim Rate (%)",
                    height=400
                )
                st.plotly_chart(fig_maint_claim, use_container_width=True)
            
            with col2:
                # Maintenance vs is_claim
                maint_is_claim = pd.crosstab(
                    balanced_data_cleaned['Need_Maintenance'],
                    balanced_data_cleaned['is_claim'],
                    normalize='index'
                ) * 100
                
                fig_maint_is_claim = go.Figure(data=[
                    go.Bar(
                        x=['No Maintenance Needed', 'Maintenance Needed'],
                        y=maint_is_claim[1],
                        name='Insurance Claim Rate',
                        marker_color='#2ecc71'
                    )
                ])
                
                fig_maint_is_claim.update_layout(
                    title="Insurance Claim Rate by Maintenance Status",
                    xaxis_title="Maintenance Status",
                    yaxis_title="Insurance Claim Rate (%)",
                    height=400
                )
                st.plotly_chart(fig_maint_is_claim, use_container_width=True)
            
            # Claim Types Comparison
            st.write("#### Claim Types Comparison")
            
            # Create correlation matrix between target variables
            target_vars = ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']
            target_corr = balanced_data_cleaned[target_vars].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=target_corr,
                x=target_vars,
                y=target_vars,
                colorscale='RdBu',
                text=np.round(target_corr, 2),
                texttemplate='%{text}',
                textfont={"size": 14},
                colorbar=dict(title='Correlation')
            ))
            
            fig_corr.update_layout(
                title="Correlation between Different Claim Types",
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Risk Factor Distribution
            st.write("#### Risk Factor Distribution Across Claim Types")
            
            # Select key risk factors
            risk_factors = ['MVR_PTS', 'OLDCLAIM', 'CLM_FREQ']
            
            # Create a combined visualization for risk factors across claim types
            risk_data = []
            
            for factor in risk_factors:
                # Calculate mean values for each target variable
                for target in target_vars:
                    mean_values = balanced_data_cleaned.groupby(target)[factor].mean()
                    for category in [0, 1]:
                        risk_data.append({
                            'Risk_Factor': factor,
                            'Target_Type': target,
                            'Category': 'Yes' if category == 1 else 'No',
                            'Mean_Value': mean_values[category]
                        })
            
            risk_df = pd.DataFrame(risk_data)
            
            # Create grouped bar chart
            fig_risk = go.Figure()
            
            for target in target_vars:
                target_data = risk_df[risk_df['Target_Type'] == target]
                fig_risk.add_trace(go.Bar(
                    name=target,
                    x=[f"{row['Risk_Factor']} ({row['Category']})" 
                       for _, row in target_data.iterrows()],
                    y=target_data['Mean_Value'],
                    text=np.round(target_data['Mean_Value'], 2),
                    textposition='auto'
                ))
            
            fig_risk.update_layout(
                title="Average Risk Factor Values by Claim Type",
                xaxis_title="Risk Factor (Category)",
                yaxis_title="Mean Value",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Key Insights
            st.write("#### Cross-Target Analysis Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Maintenance-Claim Relationships:**
                - Impact of maintenance needs on claim likelihood
                - Differences between claim types for maintained vehicles
                - Preventive maintenance effectiveness
                """)
            
            with col2:
                st.markdown("""
                **Risk Pattern Analysis:**
                - Common risk factors across claim types
                - Target variable correlations
                - Risk factor distribution patterns
                """)


    # Multivariate Analysis Tab
    with tab3:
        st.header("Multivariate Analysis")
        
        multi_tab1, multi_tab2, multi_tab3 = st.tabs([
            "3D Surface Analysis",
            "Multiple Line Slopes",
            "Relationship Patterns"
        ])
        
        # 3D Surface Analysis Tab
        with multi_tab1:
            st.subheader("3D Surface Analysis with Plane Fitting")
            
            # Variable selection
            numeric_cols = balanced_data.select_dtypes(include=['float64', 'int64']).columns
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Select X Variable", numeric_cols, index=0)
            with col2:
                y_var = st.selectbox("Select Y Variable", numeric_cols, index=1)
            with col3:
                z_var = st.selectbox("Select Z Variable", numeric_cols, index=2)
            
            # Color selection
            color_var = st.selectbox("Color by Target", ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'])
            
            # Create 3D scatter with fitted plane
            if x_var and y_var and z_var:
                # Prepare data
                x = balanced_data[x_var]
                y = balanced_data[y_var]
                z = balanced_data[z_var]
                colors = balanced_data[color_var]
                
                # Fit plane
                A = np.column_stack((x, y, np.ones_like(x)))
                coefficients, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
                
                # Create plane surface
                x_range = np.linspace(x.min(), x.max(), 10)
                y_range = np.linspace(y.min(), y.max(), 10)
                X, Y = np.meshgrid(x_range, y_range)
                Z = coefficients[0] * X + coefficients[1] * Y + coefficients[2]
                
                # Create figure
                fig = go.Figure()
                
                # Add scatter points
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Data Points'
                ))
                
                # Add fitted plane
                fig.add_trace(go.Surface(
                    x=x_range,
                    y=y_range,
                    z=Z,
                    opacity=0.7,
                    colorscale='Blues',
                    showscale=False,
                    name='Fitted Plane'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"3D Surface Analysis: {x_var} vs {y_var} vs {z_var}",
                    scene=dict(
                        xaxis_title=x_var,
                        yaxis_title=y_var,
                        zaxis_title=z_var
                    ),
                    height=700
                )
                
                # Calculate and display slopes
                slope_x = coefficients[0]
                slope_y = coefficients[1]
                intercept = coefficients[2]
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display slope analysis
                st.subheader("Slope Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Slope with respect to {x_var}:** {slope_x:.4f}")
                    st.write(f"**Slope with respect to {y_var}:** {slope_y:.4f}")
                    st.write(f"**Intercept:** {intercept:.4f}")
                
                with col2:
                    # Interpret slopes
                    st.write("**Relationship Interpretation:**")
                    
                    def interpret_slope(slope, var):
                        if abs(slope) < 0.1:
                            return f"Weak relationship with {var}"
                        elif abs(slope) < 0.5:
                            return f"Moderate relationship with {var}"
                        else:
                            return f"Strong relationship with {var}"
                    
                    st.write(interpret_slope(slope_x, x_var))
                    st.write(interpret_slope(slope_y, y_var))
                    
                    # Calculate R-squared
                    z_pred = coefficients[0] * x + coefficients[1] * y + coefficients[2]
                    r2 = 1 - (np.sum((z - z_pred) ** 2) / np.sum((z - z.mean()) ** 2))
                    st.write(f"**R-squared:** {r2:.4f}")
                
                # Add key insights
                st.subheader("Key Insights")
                insights = []
                
                # Slope direction insights
                if abs(slope_x) > abs(slope_y):
                    insights.append(f"- {x_var} has a stronger influence on {z_var} than {y_var}")
                else:
                    insights.append(f"- {y_var} has a stronger influence on {z_var} than {x_var}")
                
                # R-squared insight
                if r2 > 0.7:
                    insights.append("- The relationship is highly linear")
                elif r2 > 0.4:
                    insights.append("- The relationship shows moderate linearity")
                else:
                    insights.append("- The relationship appears to be non-linear")
                
                # Target variable insight
                avg_by_target = balanced_data.groupby(color_var)[z_var].mean()
                diff = avg_by_target.max() - avg_by_target.min()
                if diff > balanced_data[z_var].std():
                    insights.append(f"- Significant variation in {z_var} across different {color_var} groups")
                
                for insight in insights:
                    st.write(insight)
        


        # Multiple Line Slopes Analysis
        with multi_tab2:
            st.subheader("Multiple Line Slopes Analysis")
            
            # Variable selection
            numeric_cols = balanced_data.select_dtypes(include=['float64', 'int64']).columns
            
            # Select variables
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X Variable", numeric_cols, index=0, key='x_var_slopes')
            with col2:
                y_vars = st.multiselect("Select Y Variables (max 3)", numeric_cols, default=[numeric_cols[1]], key='y_vars_slopes')
            
            # Limit to 3 Y variables for clarity
            y_vars = y_vars[:3]
            
            if x_var and y_vars:
                # Create figure
                fig = go.Figure()
                
                # Color palette
                colors = ['blue', 'red', 'green']
                
                # Plot each Y variable
                for idx, y_var in enumerate(y_vars):
                    # Get data
                    x = balanced_data[x_var]
                    y = balanced_data[y_var]
                    
                    # Fit line
                    slope, intercept = np.polyfit(x, y, 1)
                    line_x = np.array([x.min(), x.max()])
                    line_y = slope * line_x + intercept
                    
                    # Calculate R²
                    y_pred = slope * x + intercept
                    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
                    
                    # Add scatter plot
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='markers',
                        name=f'{y_var} (data)',
                        marker=dict(color=colors[idx], size=5, opacity=0.5),
                        showlegend=True
                    ))
                    
                    # Add regression line
                    fig.add_trace(go.Scatter(
                        x=line_x, y=line_y,
                        mode='lines',
                        name=f'{y_var} (slope={slope:.3f}, R²={r2:.3f})',
                        line=dict(color=colors[idx], width=2),
                        showlegend=True
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Multiple Line Slopes Analysis",
                    xaxis_title=x_var,
                    yaxis_title="Values",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display slope comparisons
                st.subheader("Slope Comparisons")
                
                slope_data = []
                for y_var in y_vars:
                    slope, intercept = np.polyfit(balanced_data[x_var], balanced_data[y_var], 1)
                    y_pred = slope * balanced_data[x_var] + intercept
                    r2 = 1 - (np.sum((balanced_data[y_var] - y_pred) ** 2) / 
                             np.sum((balanced_data[y_var] - balanced_data[y_var].mean()) ** 2))
                    
                    slope_data.append({
                        'Variable': y_var,
                        'Slope': slope,
                        'R²': r2,
                        'Strength': 'Strong' if abs(slope) > 0.5 else 'Moderate' if abs(slope) > 0.1 else 'Weak',
                        'Direction': 'Positive' if slope > 0 else 'Negative'
                    })
                
                # Display as table
                st.dataframe(pd.DataFrame(slope_data))

        # Relationship Strength Visualization
        with multi_tab3:
            st.subheader("Relationship Strength Analysis")
            
            # Select target variable
            target_var = st.selectbox("Select Target Variable", 
                                    ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'],
                                    key='target_strength')
            
            # Select features for analysis
            features = st.multiselect("Select Features for Analysis",
                                    numeric_cols,
                                    default=list(numeric_cols[:5]),
                                    key='features_strength')
            
            if features:
                # Calculate correlations and slopes
                strength_data = []
                
                for feature in features:
                    # Calculate correlation
                    corr = balanced_data[feature].corr(balanced_data[target_var])
                    
                    # Calculate slope
                    slope, _ = np.polyfit(balanced_data[feature], balanced_data[target_var], 1)
                    
                    # Calculate statistical significance (t-test)
                    t_stat, p_value = stats.pearsonr(balanced_data[feature], balanced_data[target_var])
                    
                    strength_data.append({
                        'Feature': feature,
                        'Correlation': corr,
                        'Slope': slope,
                        'P-value': p_value,
                        'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
                    })
                
                # Create DataFrame
                strength_df = pd.DataFrame(strength_data)
                
                # Create heatmap of relationships
                fig = go.Figure(data=go.Heatmap(
                    z=strength_df[['Correlation']].values,
                    x=['Correlation'],
                    y=strength_df['Feature'],
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=np.round(strength_df[['Correlation']].values, 3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                ))
                
                fig.update_layout(
                    title=f"Relationship Strength with {target_var}",
                    height=400 + len(features) * 20
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed statistics
                st.subheader("Detailed Statistics")
                
                # Format DataFrame for display
                display_df = strength_df.round(3)
                display_df = display_df.sort_values('Abs_Correlation', 
                                                  ascending=False, 
                                                  key=lambda x: abs(x))
                
                st.dataframe(display_df)
                
                # Key findings
                st.subheader("Key Findings")
                
                # Strongest relationships
                strongest = display_df.iloc[0]
                st.write(f"- Strongest relationship: {strongest['Feature']} "
                        f"(correlation: {strongest['Correlation']:.3f})")
                
                # Significant relationships
                sig_features = display_df[display_df['P-value'] < 0.05]['Feature'].tolist()
                st.write(f"- Number of significant relationships: {len(sig_features)}")
                
                # Direction of relationships
                pos_count = (display_df['Correlation'] > 0).sum()
                neg_count = (display_df['Correlation'] < 0).sum()
                st.write(f"- Positive relationships: {pos_count}")
                st.write(f"- Negative relationships: {neg_count}")


def show_correlation_analysis():
    st.title("Multi-Domain Correlation Analysis")
    
    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Insurance Claims Correlations",
        "Vehicle Features Correlations", 
        "Maintenance Data Correlations",
        "Combined Analysis"
    ])
    
    # Insurance Claims Correlations Tab
    with tab1:
        st.header("Insurance Claims Correlation Analysis")
        
        # Define insurance claim features
        insurance_features = [
            'AGE', 'INCOME', 'HOME_VAL', 'YOJ', 'KIDSDRIV', 'HOMEKIDS',
            'PARENT1', 'MSTATUS', 'GENDER', 'MVR_PTS', 'CLM_FREQ', 'OLDCLAIM', 
            'CLM_AMT', 'TIF', 'URBANICITY', 'CAR_USE'
        ]
        
        # Calculate correlation matrix
        valid_features = [f for f in insurance_features if f in balanced_data.columns]
        corr_matrix = balanced_data[valid_features + ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Insurance Claims Correlation Matrix",
            height=700,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig)
        
        # Top 10 correlations
        st.subheader("Top 10 Overall Correlations")
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        stacked_corr = upper_tri.stack().reset_index()
        stacked_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        stacked_corr['Abs_Correlation'] = abs(stacked_corr['Correlation'])
        top_10 = stacked_corr.sort_values('Abs_Correlation', ascending=False).head(10)
        st.dataframe(top_10[['Feature 1', 'Feature 2', 'Correlation']].round(3))
        
        # Target variable correlations
        st.subheader("Correlations with Target Variables")
        target_var = st.selectbox("Select Target Variable", 
                                ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'],
                                key='ins_target')
        
        target_corr = corr_matrix[target_var][valid_features].sort_values(ascending=False)
        st.dataframe(target_corr.round(3))
    
    # Vehicle Features Correlations Tab
    with tab2:
        st.header("Vehicle Features Correlation Analysis")
        
        vehicle_features = [
            'CAR_AGE', 'max_power', 'max_torque', 'displacement', 'Engine_Size',
            'turning_radius', 'length', 'width', 'height', 'gross_weight',
            'Mileage', 'Fuel_Efficiency', 'cylinder', 'gear_box', 'airbags', 
            'ncap_rating', 'is_brake_assist', 'is_parking_sensors'
        ]
        
        # Similar structure as tab1 but with vehicle features
        valid_features = [f for f in vehicle_features if f in balanced_data.columns]
        corr_matrix = balanced_data[valid_features + ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Vehicle Features Correlation Matrix",
            height=700,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig)
        
        # Top 10 correlations and target correlations (similar structure as tab1)
        st.subheader("Top 10 Overall Correlations")
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        stacked_corr = upper_tri.stack().reset_index()
        stacked_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        stacked_corr['Abs_Correlation'] = abs(stacked_corr['Correlation'])
        top_10 = stacked_corr.sort_values('Abs_Correlation', ascending=False).head(10)
        st.dataframe(top_10[['Feature 1', 'Feature 2', 'Correlation']].round(3))
        
        st.subheader("Correlations with Target Variables")
        target_var = st.selectbox("Select Target Variable", 
                                ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'],
                                key='veh_target')
        target_corr = corr_matrix[target_var][valid_features].sort_values(ascending=False)
        st.dataframe(target_corr.round(3))
    
    # Maintenance Data Correlations Tab
    with tab3:
        st.header("Maintenance Data Correlation Analysis")
        
        maintenance_features = [
            'Vehicle_Age', 'Odometer_Reading', 'Service_History',
            'Maintenance_History_Code', 'Tire_Condition_Code',
            'Brake_Condition_Code', 'Battery_Status_Code',
            'Reported_Issues', 'Accident_History'
        ]
        
        # Similar structure as previous tabs
        valid_features = [f for f in maintenance_features if f in balanced_data.columns]
        corr_matrix = balanced_data[valid_features + ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']].corr()
        
        # Create heatmap and other visualizations similar to previous tabs
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Maintenance Data Correlation Matrix",
            height=700,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig)
        
        st.subheader("Top 10 Overall Correlations")
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        stacked_corr = upper_tri.stack().reset_index()
        stacked_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        stacked_corr['Abs_Correlation'] = abs(stacked_corr['Correlation'])
        top_10 = stacked_corr.sort_values('Abs_Correlation', ascending=False).head(10)
        st.dataframe(top_10[['Feature 1', 'Feature 2', 'Correlation']].round(3))
        
        st.subheader("Correlations with Target Variables")
        target_var = st.selectbox("Select Target Variable", 
                                ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'],
                                key='maint_target')
        target_corr = corr_matrix[target_var][valid_features].sort_values(ascending=False)
        st.dataframe(target_corr.round(3))
    
    # Combined Analysis Tab
    with tab4:
        st.header("Combined Correlation Analysis")
        
        # Use all features
        all_features = insurance_features + vehicle_features + maintenance_features
        valid_features = [f for f in all_features if f in balanced_data.columns]
        corr_matrix = balanced_data[valid_features + ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']].corr()
        
        # Create complete heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title="Combined Correlation Matrix",
            height=900,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig)
        
        st.subheader("Top 10 Overall Correlations")
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        stacked_corr = upper_tri.stack().reset_index()
        stacked_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        stacked_corr['Abs_Correlation'] = abs(stacked_corr['Correlation'])
        top_10 = stacked_corr.sort_values('Abs_Correlation', ascending=False).head(10)
        st.dataframe(top_10[['Feature 1', 'Feature 2', 'Correlation']].round(3))
        
        st.subheader("Correlations with Target Variables")
        target_var = st.selectbox("Select Target Variable", 
                                ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim'],
                                key='comb_target')
        target_corr = corr_matrix[target_var][valid_features].sort_values(ascending=False)
        st.dataframe(target_corr.round(3))

def show_dimensionality_reduction():
    st.title("Dimensionality Reduction Analysis")
    
    # Create tabs for different domains
    domain_tab1, domain_tab2, domain_tab3 = st.tabs([
        "Insurance Claims PCA",
        "Vehicle Features PCA",
        "Maintenance Data PCA"
    ])
    
    # Define feature groups for each domain
    insurance_features = [
        'CLM_AMT', 'CLM_FREQ', 'OLDCLAIM', 'HOME_VAL', 'MSTATUS',
        'AGE', 'HOMEKIDS', 'INCOME', 'YOJ', 'MVR_PTS'
    ]
    
    vehicle_features = [
        'max_power', 'length', 'width', 'displacement', 'turning_radius',
        'Engine_Size', 'max_torque', 'Mileage', 'height', 'gross_weight'
    ]
    
    maintenance_features = [
        'Service_History', 'Maintenance_History_Code', 'Reported_Issues',
        'Battery_Status_Code', 'Tire_Condition_Code', 'Brake_Condition_Code',
        'Vehicle_Age', 'Odometer_Reading', 'Fuel_Efficiency'
    ]
    
    # Insurance Claims PCA
    with domain_tab1:
        st.subheader("Insurance Claims Domain PCA")
        
        # Filter valid features
        valid_insurance = [f for f in insurance_features if f in balanced_data.columns]
        
        if valid_insurance:
            # Standardize features
            scaler = StandardScaler()
            insurance_scaled = scaler.fit_transform(balanced_data[valid_insurance])
            
            # Apply PCA
            pca_insurance = PCA()
            insurance_pca = pca_insurance.fit_transform(insurance_scaled)
            
            # Scree plot
            exp_var_ratio = pca_insurance.explained_variance_ratio_
            
            fig_scree = go.Figure(data=[
                go.Bar(
                    x=[f"PC{i+1}" for i in range(len(exp_var_ratio))],
                    y=exp_var_ratio,
                    text=np.round(exp_var_ratio * 100, 1),
                    textposition='auto'
                )
            ])
            
            fig_scree.update_layout(
                title="Explained Variance Ratio by Principal Component",
                xaxis_title="Principal Components",
                yaxis_title="Explained Variance Ratio",
                showlegend=False
            )
            
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # Feature contributions
            loadings = pca_insurance.components_.T
            pc_df = pd.DataFrame(
                loadings,
                columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
                index=valid_insurance
            )
            
            fig_loadings = go.Figure(data=go.Heatmap(
                z=loadings,
                x=[f'PC{i+1}' for i in range(loadings.shape[1])],
                y=valid_insurance,
                colorscale='RdBu',
                text=np.round(loadings, 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_loadings.update_layout(
                title="Feature Loadings on Principal Components",
                height=600
            )
            
            st.plotly_chart(fig_loadings, use_container_width=True)
            
            # 2D PCA plot with target variables
            for target in ['CLAIM_FLAG', 'Need_Maintenance', 'is_claim']:
                if target in balanced_data.columns:
                    fig_2d = go.Figure()
                    
                    for class_val in [0, 1]:
                        mask = balanced_data[target] == class_val
                        fig_2d.add_trace(go.Scatter(
                            x=insurance_pca[mask, 0],
                            y=insurance_pca[mask, 1],
                            mode='markers',
                            name=f'{target}={class_val}',
                            opacity=0.7
                        ))
                    
                    fig_2d.update_layout(
                        title=f"2D PCA Plot Colored by {target}",
                        xaxis_title="First Principal Component",
                        yaxis_title="Second Principal Component",
                        height=500
                    )
                    
                    st.plotly_chart(fig_2d, use_container_width=True)
    
    # Vehicle Features PCA
    with domain_tab2:
        st.subheader("Vehicle Features Domain PCA")
        # [Similar implementation for vehicle features]
        valid_vehicle = [f for f in vehicle_features if f in balanced_data.columns]
        
        if valid_vehicle:
            # Standardize features
            scaler = StandardScaler()
            vehicle_scaled = scaler.fit_transform(balanced_data[valid_vehicle])
            
            # Apply PCA
            pca_vehicle = PCA()
            vehicle_pca = pca_vehicle.fit_transform(vehicle_scaled)
            
            # [Similar visualizations as insurance domain]
            # Implementation continues...
    
    # Maintenance Data PCA
    with domain_tab3:
        st.subheader("Maintenance Data Domain PCA")
        # [Similar implementation for maintenance features]
        valid_maintenance = [f for f in maintenance_features if f in balanced_data.columns]
        
        if valid_maintenance:
            # Standardize features
            scaler = StandardScaler()
            maintenance_scaled = scaler.fit_transform(balanced_data[valid_maintenance])
            
            # Apply PCA
            pca_maintenance = PCA()
            maintenance_pca = pca_maintenance.fit_transform(maintenance_scaled)
            
            # [Similar visualizations as insurance domain]
            # Implementation continues...
    
    # Add insights section
    st.subheader("Key Dimensionality Reduction Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Domain-Specific Patterns:**
        - Insurance: Key components and their interpretations
        - Vehicle: Technical specification reduction
        - Maintenance: Combined condition indicators
        """)
    
    with col2:
        st.markdown("""
        **Recommendations:**
        - Optimal number of components for each domain
        - Feature selection strategies
        - Composite feature suggestions
        """)

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
    elif selected_page == "Data Missingness Analysis":
        show_missingness_analysis()
    elif selected_page == "Data Transformations":
        show_data_transformations()
    elif selected_page == "Data Merging & Integration":
        show_data_merging()
    elif selected_page == "EDA":
        show_eda()
    elif selected_page == "Correlation Analysis":
        show_correlation_analysis()
    elif selected_page == "Dimensionality Reduction":
        show_dimensionality_reduction()
elif selected_space == "Production Space":
    if selected_page == "Risk Assessment":
        show_risk_assessment()
    elif selected_page == "Vehicle Comparison":
        show_vehicle_comparison()
    elif selected_page == "Maintenance Predictor":
        show_maintenance_predictor()
    elif selected_page == "Insurance Calculator":
        show_insurance_calculator()


