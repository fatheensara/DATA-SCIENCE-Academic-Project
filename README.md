# Predictive-Health-Assessment-for-Safe-Fasting-During-Ramadan

## 1.0 INTRODUCTION

This project focuses on developing a predictive model to determine if individuals with chronic medical conditions can fast safely during Ramadan. Using a dataset of over 100,000 medical records, we utilized the **Random Forest** algorithm and **Synthetic Minority Over-sampling Technique (SMOTE)** to provide evidence-based, individualized medical advice. The system addresses a critical gap in healthcare where general guidelines often fail to account for specific physiological risks like hypoglycemia, dehydration, and metabolic stress. By analyzing 63 clinical, demographic, and anthropometric variables, the model offers a data-driven solution to help patients reconcile religious commitments with medical safety.

## 2.0 OBJECTIVES

***

* **Identify Correlations**: To identify the relationship between certain medical conditions (e.g., chronic liver disease, immune-compromised status) and a Muslim's ability to fast.


* **Analyze Biomarkers**: To analyze the impact of clinical biomarkers such as blood sugar levels and blood pressure on fasting endurance.


* **Assess Demographics**: To assess the influence of demographic factors (age, gender, BMI, and ethnicity) on an individual's fasting capability.



## 3.0 DATA SOURCES

---

* **Dataset**: "Ability to Fast Ramadan Classification" sourced from Kaggle.


* **Volume**: 104,125 records and 63 features.


* **Features**: Includes demographic variables (age, gender, ethnicity), anthropometric data (weight, height, BMI), and clinical observations such as glucose, albumin, and bilirubin levels.



## 4.0 PREREQUISITES

---

Before running the analysis, ensure the following steps are completed:

* **Environment**: Install a Python 3.12 environment (Anaconda or Google Colab recommended).


* **Library Installation**: Install the necessary data science libraries:


`pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy`.


* **Dataset Placement**: Ensure the file `data.csv` is in the same directory as the notebooks.


* **Data Preprocessing**: The scripts automatically handle:
* **Standardization**: Cleaning `gender_identity` and `liver_function_impaired` into binary values.


* **Imputation**: Automated **KNN imputation** for missing values across all clinical and demographic columns.


* **Feature Engineering**: Creating BMI categories and segmenting age into four life stages.





## 5.0 RUNNING THE SCRIPT

---

The methodology follows a structured data science pipeline implemented in Python:

* **Data Acquisition**: Load and explore the dataset using the `pandas` library.


* **Preprocessing**: Perform label encoding for categorical variables and handle missing values using median/mode or KNN imputation.


* **Data Balancing**: Apply **SMOTE** to address the class imbalance, as 78.4% of the original population could fast while only 21.6% could not.


* **Model Development**: Train a **Random Forest Classifier** with an 80/20 train-test split.


* **Hyperparameter Tuning**: Optimize the model using `RandomizedSearchCV` with cross-validation.



## 6.0 EVALUATION

---

The model was evaluated using a comprehensive suite of performance metrics:

* **AUC-ROC**: The model achieved an area under the curve (AUC) of **0.8226**, indicating high discriminatory ability.


* **Accuracy**: The final model achieved a benchmark accuracy of **80.80%**.


* **Feature Importance**: Blood glucose levels (`OBS1_glucose_max`) were identified as the most critical predictor, followed by BMI and age.


* **Comparative Analysis**: The clinical-only model achieved 78.42% accuracy, while the demographic-only model reached 72.20%.



## 7.0 REQUIREMENTS

---

This project requires Python 3.x and the following libraries:

* `pandas` & `numpy` (Data manipulation) 


* `scikit-learn` (Preprocessing and modeling) 

* `matplotlib` & `seaborn` (Visualization) 


* `imblearn` (SMOTE implementation) 


* `scipy.stats` (Statistical testing) 


### 7.0 TEAM
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
