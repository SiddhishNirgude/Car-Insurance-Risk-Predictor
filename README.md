# Michigan State University  
CMSE-830 Class Project  

## Car Insurance Risk Predictor using  
Machine Learning and Data Analytics  

### By - Siddhish Nirgude  

This repository contains a comprehensive project exploring car insurance risk prediction using advanced machine learning models and data analytics.

![Car Insurance Image](https://github.com/SiddhishNirgude/Car-Insurance-Risk-Predictor/blob/main/images/Imageof-Auto-Insurance.jpg)

---------------------------------
INDEX
---------------------------------
1. Repository Contents
2. Data Pipeline
3. Project Implementation
4. References

---------------------------------
Repository Contents
---------------------------------
### Main Files
- `Car Insurance and Features.ipynb`: Main Jupyter notebook containing complete data analysis and modeling
- `streamlit_final.py`: Interactive web application for production deployment

### Original Datasets
- `Vehicle_features_data.csv`: Car specifications and features dataset
- `car_insurance_claim.csv`: Insurance claim history and details
- `vehicle_maintenance_data.csv`: Vehicle maintenance records and history

### Data Processing Pipeline

1. **Data Cleaning Stage**
   - `insurance_clean.csv`: Cleaned insurance dataset
   - `features_clean.csv`: Cleaned vehicle features dataset
   - `maintenance_clean.csv`: Cleaned maintenance dataset

2. **Missing Value Analysis & Imputation**
   - `features_missing.csv`: Dataset with missingness analysis
   - `features_after_imputation.csv`: Features data after handling missing values
   - `insurance_after_imputation.csv`: Insurance data after imputation
   - `maintenance_after_imputation.csv`: Maintenance data after imputation

3. **Encoding Stage**
   - `features_encoded.csv`: Encoded categorical features
   - `insurance_encoded.csv`: Encoded insurance data
   - `maintenance_encoded.csv`: Encoded maintenance data

4. **Transformation Stage**
   - `insurance_final_df.csv`: Transformed insurance data (preserved rows due to moderate size)
   - `features_final_reduced.csv`: Reduced feature dataset (rows removed due to large size)
   - `maintenance_final_reduced.csv`: Reduced maintenance dataset (rows removed due to large size)

5. **Integration Stage**
   - `final_integrated_dataset.csv`: Initial integrated dataset
   - `final_integrated_df_cleaned.csv`: Cleaned version of integrated dataset
   - `final_integrated_no_duplicates.csv`: Integrated data after duplicate removal
   - `final_integrated_df_cleaned_balanced_3.csv`: Final balanced dataset used for modeling

----------------------------------
Data Pipeline
----------------------------------

### 1. Data Collection
- Vehicle features data from Kaggle's Car Insurance Claim Prediction dataset
- Insurance claim data from Car Insurance Claim dataset
- Maintenance records from Vehicle Maintenance dataset

### 2. Data Preprocessing
1. **Cleaning Phase**
   - Handling missing values
   - Removing duplicates
   - Data type corrections
   - Format standardization

2. **Feature Engineering**
   - Missing value analysis
   - Data imputation
   - Categorical encoding
   - Feature transformation

3. **Data Integration**
   - Merging datasets
   - Post-integration cleaning
   - Duplicate removal
   - Class balancing

----------------------------------
References
----------------------------------

**Data Sources:**
- Vehicle Features Data: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification?select=train.csv
- Insurance Claims Data: https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data
- Vehicle Maintenance Data: https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data

**Technical Documentation:**
- Streamlit: https://docs.streamlit.io/
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/docs/
- Python: https://docs.python.org/

**Notes:**
- The original datasets were processed through multiple stages of cleaning and transformation
- Some datasets were reduced in size due to computational constraints
- The final balanced dataset was used for model training and evaluation
