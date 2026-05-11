# Predictive-Health-Assessment-for-Safe-Fasting-During-Ramadan

This project develops a data-driven predictive model to evaluate the medical safety of fasting during Ramadan for individuals with chronic health conditions. By leveraging machine learning, the system identifies physiological risks such as hypoglycemia and metabolic stress, providing evidence-based, individualized guidance to help patients reconcile religious practices with medical safety .



### 1.0 PROJECT OVERVIEW
***
Generic medical advice often fails to account for the specific risks faced by people with diverse health histories during the month-long fast of Ramadan. This project addresses that gap by analyzing a complex dataset of over 100,000 medical records using advanced classification techniques.

#### Key Objectives

* **Medical Correlation**: Identify the relationship between severe clinical biomarkers (e.g., liver failure, immune-compromised status) and fasting capability .


* **Biomarker Analysis**: Evaluate the impact of blood sugar regulation and blood pressure on fasting endurance.


* **Demographic Assessment**: Determine the predictive reliability of age, gender, BMI, and ethnicity on an individual's health status during the fast .


### 2.0 DATASET & PREPROCESSING
***
The project utilizes the **"Ability to Fast Ramadan Classification"** dataset.

#### Data Profile

* **Volume**: 104,125 records across 63 features.


* **Initial Distribution**: The raw data showed a significant class imbalance: 78.4% "Can Fast" (81,618 samples) vs. 21.6% "Cannot Fast" (22,507 samples) .


* **Missing Values**: Second-phase lab observations (OBS2) exhibited material deficiencies, with bilirubin and albumin markers absent in over 90% of instances .



#### Pipeline Steps

1. **Standardization**: Categorical variables like `gender_identity` and clinical conditions were standardized into binary values .


2. **Imputation**: Automated **KNN imputation** was used to address missing clinical and demographic values.


3. **Balancing**: The **Synthetic Minority Over-sampling Technique (SMOTE)** was applied to create a balanced training set of 130,588 total samples (50/50 split) .



### 3.0 MODEL DEVELOPMENT & EVALUATION
***
The **Random Forest Classifier** was selected for its resilience in managing high-dimensional medical data and non-linear relationships.

#### Performance Metrics

The model demonstrated high discriminatory ability and reliability:

| Metric | Score |
| --- | --- |
| **Accuracy** | 80.80% 

 
| **ROC-AUC** | 0.8226 

 
| **Weighted F1-Score** | 0.8078 

 
| **Mean CV F1-Score** | 0.7890 (±0.0015) 

 

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \times 100\%$$



#### Feature Importance Ranking

Feature importance analysis highlighted that metabolic status is more influential than chronic disease status alone within this dataset :

1. **OBS1_glucose_max**: 0.1419 


2. **OBS2_glucose_max**: 0.0480 


3. **Body Mass Index (BMI)**: 0.0383 


4. **Age (yrs)**: 0.0302 



### 4.0 KEY FINDINGS
***
* **Medical Factors (RQ1)**: A significant relationship was found between liver function impairment and fasting ability ($p = 0.0086$), though its predictive weight in the model was lower than expected (rank 60, score 0.0004) .


* **Demographic Reliability (RQ2)**: Basic demographics (Age, BMI, Weight) are **moderately reliable** predictors, achieving 72.23% accuracy, but are less effective than clinical biomarkers like glucose levels.





### 5.0 PREREQUISITES
***
* **Environment**: Python 3.12.
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `imblearn`, `matplotlib`, `seaborn`, `scipy` .


* **Setup**: Place `data.csv` in the root directory. The scripts handle KNN imputation and label encoding automatically.



### 6.0 TEAM
***
* **Aliah Maisarah Binti Roslee** 


* **Galeya Binti Herman Gallego** 


* **Aida Binti Shahiedun** 

* **Muhammad Izzat Bin Zamri** 


* **Fatheen Sara Sofiah Binti Romy Norfidzy** 



### ***📝 AUTHOR***
***
***Fatheen Sara Sofiah binti Romy Norfidzy***

***This project is for educational purposes.***
