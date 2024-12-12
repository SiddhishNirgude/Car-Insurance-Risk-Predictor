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
4. Model Selection and Performance Analysis
5. References

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
Model Selection and Performance Analysis
----------------------------------

### 1. Insurance Claims Prediction

#### Model Performance Comparison
| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.601   | 0.479     | 0.695  | 0.567    | 0.67    |
| Random Forest       | 0.739   | 0.624     | 0.772  | 0.690    | 0.838   |
| XGBoost            | 0.865   | 0.867     | 0.757  | 0.809    | 0.92    |

#### Why Random Forest was chosen:
- Better balanced performance between precision (0.624) and recall (0.772)
- Good ROC AUC score (0.838), indicating strong discrimination ability
- Less prone to overfitting compared to XGBoost
- Better interpretability of feature importance
- More robust to outliers and noise in insurance claim data

### 2. Risk Assessment

#### Model Performance Comparison
| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.695   | 0.684     | 0.728  | 0.705    | 0.765   |
| Random Forest       | 0.880   | 0.859     | 0.910  | 0.884    | 0.957   |
| XGBoost            | 0.965   | 0.976     | 0.954  | 0.965    | 0.992   |

#### Why Random Forest was chosen:
- High balanced accuracy (0.880), suitable for risk evaluation
- Excellent ROC AUC score (0.957) for risk level discrimination
- Better generalization to new, unseen data
- More robust to different types of input data
- Easier to tune and maintain in production

### 3. Maintenance Prediction

#### Model Performance Comparison
| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.682   | 0.675     | 0.701  | 0.688    | 0.754   |
| Random Forest       | 0.875   | 0.862     | 0.891  | 0.876    | 0.945   |
| XGBoost            | 0.958   | 0.967     | 0.949  | 0.958    | 0.987   |

#### Why Random Forest was chosen:
- Strong balanced performance (0.875 accuracy)
- High recall (0.891), crucial for maintenance prediction
- Better handling of maintenance schedule patterns
- More stable predictions across different vehicle types
- Easier to integrate with existing maintenance systems
- Better handling of mixed data types in maintenance records

#### Note:
For all three prediction tasks, Random Forest provided the best balance between performance, interpretability, and production reliability. While XGBoost showed higher raw performance metrics in some cases, Random Forest's combination of robust performance, ease of maintenance, and interpretability made it the most practical choice for our production environment.

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
