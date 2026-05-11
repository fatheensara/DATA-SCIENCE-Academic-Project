import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, roc_curve, 
                            average_precision_score, f1_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
def load_and_explore_data(filepath):
    print("="*80)
    print("DATA LOADING AND EXPLORATION")
    print("="*80)
    
    df = pd.read_csv(filepath, header=0)
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False))
    
    # Target variable distribution
    print("\nTarget variable distribution:")
    print(df['fasting_ability'].value_counts(normalize=True) * 100)
    
    # Create a pie chart for target distribution
    plt.figure(figsize=(10, 6))
    df['fasting_ability'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, 
                                                colors=['#ff9999','#66b3ff'])
    plt.title('Distribution of Fasting Ability', fontsize=16)
    plt.ylabel('')
    plt.savefig('target_distribution.png')
    plt.close()
    
    return df

# Preprocess the data
def preprocess_data(df):
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop the ID column if it exists
    df_processed.drop('id', axis=1, inplace=True, errors='ignore')
    
    # Clean up gender_identity - more robust handling
    def clean_gender(x):
        if pd.isna(x):
            return np.nan
        elif str(x).lower() in ['male', 'm']:
            return 0
        elif str(x).lower() in ['female', 'f']:
            return 1
        else:
            return np.nan
    
    # Clean up boolean columns
    def clean_boolean(x):
        if pd.isna(x):
            return np.nan
        elif str(x).lower() in ['true', 't', 'yes', 'y', '1']:
            return 1
        elif str(x).lower() in ['false', 'f', 'no', 'n', '0']:
            return 0
        else:
            return np.nan
    
    # Apply cleaning functions
    if 'gender_identity' in df_processed.columns:
        df_processed['gender_identity'] = df_processed['gender_identity'].apply(clean_gender)
        print("\nUnique values in gender_identity after cleaning:", df_processed['gender_identity'].unique())
    
    # Apply boolean cleaning to relevant columns
    boolean_columns = ['liver_function_impaired']
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(clean_boolean)
            print(f"\nUnique values in {col} after cleaning:", df_processed[col].unique())
    
    # Encode categorical variables
    if 'ethnic_group' in df_processed.columns:
        print("\nEncoding ethnic_group...")
        encoder = LabelEncoder()
        df_processed['ethnic_group'] = encoder.fit_transform(df_processed['ethnic_group'].astype(str))
        print("Unique encoded values:", df_processed['ethnic_group'].unique())
    
    # Feature engineering - create new features
    print("\nPerforming feature engineering...")
    
    # Create BMI categories
    if 'body_mass_index' in df_processed.columns:
        df_processed['bmi_category'] = pd.cut(
            df_processed['body_mass_index'], 
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
        )
        print("Created BMI categories")
    
    # Create age groups if age is present
    if 'age' in df_processed.columns:
        df_processed['age_group'] = pd.cut(
            df_processed['age'],
            bins=[0, 18, 40, 65, 120],
            labels=[0, 1, 2, 3]  # Child, Adult, Middle-aged, Senior
        )
        print("Created age groups")
    
    # Add visualization - correlation heatmap for all numeric columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:  # At least 2 columns needed for correlation
        plt.figure(figsize=(14, 12))
        corr_matrix = df_processed[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
    
    # Visualize BMI distribution by fasting ability if present
    if 'body_mass_index' in df_processed.columns and 'fasting_ability' in df_processed.columns:
        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=df_processed, x='body_mass_index', hue='fasting_ability', 
                    fill=True, common_norm=False, alpha=0.5)
        plt.title('Distribution of BMI by Fasting Ability', fontsize=16)
        plt.xlabel('Body Mass Index')
        plt.ylabel('Density')
        plt.savefig('bmi_by_fasting_ability.png')
        plt.close()
        print("\nBMI distribution by fasting ability saved as 'bmi_by_fasting_ability.png'")
    
    # Visualize age distribution by fasting ability if present
    if 'age' in df_processed.columns and 'fasting_ability' in df_processed.columns:
        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=df_processed, x='age', hue='fasting_ability', 
                   fill=True, common_norm=False, alpha=0.5)
        plt.title('Distribution of Age by Fasting Ability', fontsize=16)
        plt.xlabel('Age')
        plt.ylabel('Density')
        plt.savefig('age_by_fasting_ability.png')
        plt.close()
        print("\nAge distribution by fasting ability saved as 'age_by_fasting_ability.png'")
    
    # Impute missing values with SimpleImputer
    print("\nImputing missing values...")
    
    # First, identify columns with missing values
    cols_with_missing = df_processed.columns[df_processed.isnull().any()].tolist()
    if cols_with_missing:
        print(f"Columns with missing values: {cols_with_missing}")
        
        # Simple imputation - use median for numeric columns and most frequent for categorical
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(exclude=['int64', 'float64']).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
        
        # Impute categorical columns with most frequent value
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
        
        # Check if imputation worked
        print("Missing values after imputation:", df_processed.isnull().sum().sum())
        
        return df_processed
    else:
        print("No missing values to impute")
        return df_processed

# Analyze feature importance 
def analyze_feature_importance(X, y, max_features=15):
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Initialize model with fewer trees for speed
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature ranking:")
    for i, idx in enumerate(indices[:max_features]):  # Print top features
        print(f"{i+1}. {X.columns[idx]} ({importances[idx]:.4f})")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance for Fasting Ability Prediction', fontsize=16)
    plt.bar(range(len(indices[:max_features])), importances[indices[:max_features]], align='center')
    plt.xticks(range(len(indices[:max_features])), [X.columns[i] for i in indices[:max_features]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Add a horizontal bar plot for better visualization
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(indices[:max_features])), importances[indices[:max_features]], align='center')
    plt.yticks(range(len(indices[:max_features])), [X.columns[i] for i in indices[:max_features]])
    plt.title('Feature Importance (Horizontal Bar Plot)', fontsize=16)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_horizontal.png')
    plt.close()
    
    return model, indices

# Train and evaluate model
def train_and_evaluate_model(X, y, reduced_hyperparams=True):
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    # Check for class imbalance
    print("\nClass distribution:")
    class_counts = np.bincount(y)
    print(f"Class 0 (No Fasting): {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Class 1 (Fasting): {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution', fontsize=16)
    plt.xticks([0, 1], ['No Fasting', 'Fasting'])
    plt.savefig('class_distribution.png')
    plt.close()
    print("\nClass distribution visualization saved as 'class_distribution.png'")
    
    # Check if class imbalance needs to be addressed (if one class is less than 30% of data)
    imbalance_ratio = min(class_counts) / max(class_counts)
    use_smote = imbalance_ratio < 0.3
    
    if use_smote:
        print("\nApplying SMOTE to address class imbalance...")
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to training data if needed
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:")
        train_class_counts = np.bincount(y_train)
        print(f"Class 0 (No Fasting): {train_class_counts[0]} ({train_class_counts[0]/len(y_train)*100:.2f}%)")
        print(f"Class 1 (Fasting): {train_class_counts[1]} ({train_class_counts[1]/len(y_train)*100:.2f}%)")
        
        # Visualize class distribution after SMOTE
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y_train)
        plt.title('Class Distribution After SMOTE', fontsize=16)
        plt.xticks([0, 1], ['No Fasting', 'Fasting'])
        plt.savefig('class_distribution_after_smote.png')
        plt.close()
        print("\nClass distribution after SMOTE saved as 'class_distribution_after_smote.png'")
    
    # Hyperparameter tuning - reduced search space for speed
    print("\nPerforming hyperparameter tuning...")
    
    if reduced_hyperparams:
        # Reduced parameter grid for faster execution
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 20],
            'min_samples_split': [2, 10],
            'class_weight': [None, 'balanced']
        }
        # Fewer iterations
        n_iter = 8
    else:
        # Original parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
        n_iter = 20
    
    # Random search with cross-validation
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    rs = RandomizedSearchCV(
        rf, param_grid, n_iter=n_iter, cv=cv, scoring='f1_weighted',
        random_state=42, n_jobs=-1, verbose=0
    )
    
    rs.fit(X_train, y_train)
    
    print(f"\nBest parameters: {rs.best_params_}")
    best_model = rs.best_estimator_
    
    # Fit the model with best parameters
    best_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Print comprehensive evaluation
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fasting', 'Fasting'], 
                yticklabels=['No Fasting', 'Fasting'])
    plt.title('Confusion Matrix for Fasting Ability Prediction', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, marker='.', label=f'Random Forest (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Fasting Ability Prediction', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Add Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', label=f'Random Forest (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Fasting Ability Prediction', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    return best_model, X_test, y_test

# Medical analysis function - enhanced with visualizations
def analyze_medical_factors(df, model):
    print("\n" + "="*80)
    print("MEDICAL FACTORS ANALYSIS (RESEARCH QUESTION 1)")
    print("="*80)
    
    # Identify medical columns
    medical_columns = [
        'liver_function_impaired',
        # Add more medical columns as needed
    ]
    
    # Filter only existing columns
    medical_columns = [col for col in medical_columns if col in df.columns]
    
    if not medical_columns:
        print("No medical columns found in the dataset")
        return
    
    print("\nAnalyzing relationship between medical conditions and fasting ability...")
    
    # Calculate correlation with target
    corr_with_target = {}
    for col in medical_columns:
        if df[col].dtype in ['int64', 'float64']:
            corr = df[col].corr(df['fasting_ability'])
            corr_with_target[col] = corr
    
    # Print correlation results
    if corr_with_target:
        print("\nCorrelation of medical conditions with fasting ability:")
        for col, corr in sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{col}: {corr:.4f}")
    
    # Add heatmap visualization for medical factors
    if len(medical_columns) > 0:
        # Create correlation heatmap
        med_columns = medical_columns + ['fasting_ability']
        plt.figure(figsize=(10, 8))
        corr_matrix = df[med_columns].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                    vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Correlation Heatmap: Medical Factors vs Fasting Ability', fontsize=16)
        plt.tight_layout()
        plt.savefig('medical_correlation_heatmap.png')
        plt.close()
        print("\nMedical factors correlation heatmap saved as 'medical_correlation_heatmap.png'")
    
    # Focus on liver function impairment
    if 'liver_function_impaired' in df.columns:
        print("\nSubgroup analysis for patients with liver function impairment:")
        liver_patients = df[df['liver_function_impaired'] == 1]
        print(f"Number of patients with liver function impairment: {len(liver_patients)}")
        print(f"Percentage able to fast: {liver_patients['fasting_ability'].mean() * 100:.2f}%")
        
        # Compare with general population
        print(f"General population percentage able to fast: {df['fasting_ability'].mean() * 100:.2f}%")
        
        # Create visualization comparing liver function impairment and fasting ability
        plt.figure(figsize=(12, 8))
        
        # Create a contingency table
        contingency = pd.crosstab(df['liver_function_impaired'], df['fasting_ability'])
        contingency_percent = contingency.div(contingency.sum(axis=1), axis=0) * 100
        
        # Plot stacked bar chart
        contingency_percent.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Fasting Ability by Liver Function Impairment Status', fontsize=16)
        plt.xlabel('Liver Function Impaired')
        plt.ylabel('Percentage')
        plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
        plt.legend(['Cannot Fast', 'Can Fast'])
        plt.savefig('liver_function_fasting_ability.png')
        plt.close()
        print("\nLiver function vs fasting ability visualization saved as 'liver_function_fasting_ability.png'")
        
        # Statistical comparison
        from scipy.stats import chi2_contingency
        
        contingency = pd.crosstab(df['liver_function_impaired'], df['fasting_ability'])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        print(f"Chi-square test: chi2 = {chi2:.4f}, p-value = {p:.4f}")
        print("Conclusion: " + ("Significant relationship" if p < 0.05 else "No significant relationship") + 
              " between liver function impairment and fasting ability")
    
    # Build a model focused on medical factors only
    print("\nBuilding a model using only medical variables...")
    if len(medical_columns) > 0:
        X_med = df[medical_columns]
        y_med = df['fasting_ability']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_med, y_med, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model_med = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_med.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model_med.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Medical-only model accuracy: {accuracy:.4f}")
        print("\nClassification Report (Medical-only model):")
        print(classification_report(y_test, y_pred))
        
        # Visualize confusion matrix for medical model
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fasting', 'Fasting'], 
                    yticklabels=['No Fasting', 'Fasting'])
        plt.title('Confusion Matrix for Medical-Only Model', fontsize=16)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('medical_confusion_matrix.png')
        plt.close()
        print("\nMedical model confusion matrix saved as 'medical_confusion_matrix.png'")
        
        # Feature importance for medical model
        importances = model_med.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Medical Factor Importance for Fasting Ability Prediction', fontsize=16)
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [X_med.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('medical_feature_importance.png')
        plt.close()
        print("\nMedical feature importance saved as 'medical_feature_importance.png'")

# Demographic analysis function - enhanced with visualizations
def analyze_demographic_factors(df, model):
    print("\n" + "="*80)
    print("DEMOGRAPHIC FACTORS ANALYSIS (RESEARCH QUESTION 2)")
    print("="*80)
    
    # Identify demographic columns
    demographic_columns = [
        'gender_identity',
        'body_mass_index',
        'age',
        'bmi_category',
        'age_group'
    ]
    
    # Filter only existing columns
    demographic_columns = [col for col in demographic_columns if col in df.columns]
    
    if not demographic_columns:
        print("No demographic columns found in the dataset")
        return
    
    print("\nAnalyzing relationship between demographic factors and fasting ability...")
    
    # Calculate correlation with target for numerical variables
    corr_with_target = {}
    for col in demographic_columns:
        if df[col].dtype in ['int64', 'float64']:
            corr = df[col].corr(df['fasting_ability'])
            corr_with_target[col] = corr
    
    # Print correlation results
    if corr_with_target:
        print("\nCorrelation of demographic factors with fasting ability:")
        for col, corr in sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{col}: {corr:.4f}")
    
    # Add correlation heatmap for demographic factors
    if len(demographic_columns) > 0:  
        demo_columns = demographic_columns + ['fasting_ability']
        numeric_demo_cols = df[demo_columns].select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_demo_cols) > 1:  # Need at least 2 numeric columns for correlation
            plt.figure(figsize=(12, 10))
            corr_matrix = df[numeric_demo_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                        linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Correlation Heatmap: Demographic Factors', fontsize=16)
            plt.tight_layout()
            plt.savefig('demographic_correlation_heatmap.png')
            plt.close()
            print("\nDemographic factors correlation heatmap saved as 'demographic_correlation_heatmap.png'")
    
    # Visualize distribution of fasting ability by gender if available
    if 'gender_identity' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='gender_identity', hue='fasting_ability', data=df)
        plt.title('Fasting Ability by Gender', fontsize=16)
        plt.xlabel('Gender (0=Male, 1=Female)')
        plt.xticks([0, 1], ['Male', 'Female'])
        plt.ylabel('Count')
        plt.legend(['Cannot Fast', 'Can Fast'])
        plt.savefig('gender_fasting_ability.png')
        plt.close()
        print("\nGender vs fasting ability visualization saved as 'gender_fasting_ability.png'")
    
    # Visualize distribution of fasting ability by BMI category if available
    if 'bmi_category' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(x='bmi_category', hue='fasting_ability', data=df)
        plt.title('Fasting Ability by BMI Category', fontsize=16)
        plt.xlabel('BMI Category (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)')
        plt.xticks([0, 1, 2, 3], ['Underweight', 'Normal', 'Overweight', 'Obese'])
        plt.ylabel('Count')
        plt.legend(['Cannot Fast', 'Can Fast'])
        plt.savefig('bmi_category_fasting_ability.png')
        plt.close()
        print("\nBMI category vs fasting ability visualization saved as 'bmi_category_fasting_ability.png'")
    
    # Visualize distribution of fasting ability by age group if available
    if 'age_group' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(x='age_group', hue='fasting_ability', data=df)
        plt.title('Fasting Ability by Age Group', fontsize=16)
        plt.xlabel('Age Group (0=Child, 1=Adult, 2=Middle-aged, 3=Senior)')
                # Continue from where the code left off in analyze_demographic_factors()
        plt.xticks([0, 1, 2, 3], ['Child', 'Adult', 'Middle-aged', 'Senior'])
        plt.ylabel('Count')
        plt.legend(['Cannot Fast', 'Can Fast'])
        plt.savefig('age_group_fasting_ability.png')
        plt.close()
        print("\nAge group vs fasting ability visualization saved as 'age_group_fasting_ability.png'")
    
    # Build a model focused on demographic factors only
    print("\nBuilding a model using only demographic variables...")
    if len(demographic_columns) > 0:
        X_demo = df[demographic_columns]
        y_demo = df['fasting_ability']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_demo, y_demo, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model_demo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_demo.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model_demo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Demographic-only model accuracy: {accuracy:.4f}")
        print("\nClassification Report (Demographic-only model):")
        print(classification_report(y_test, y_pred))
        
        # Visualize confusion matrix for demographic model
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fasting', 'Fasting'], 
                    yticklabels=['No Fasting', 'Fasting'])
        plt.title('Confusion Matrix for Demographic-Only Model', fontsize=16)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('demographic_confusion_matrix.png')
        plt.close()
        print("\nDemographic model confusion matrix saved as 'demographic_confusion_matrix.png'")
        
        # Feature importance for demographic model
        importances = model_demo.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Demographic Factor Importance for Fasting Ability Prediction', fontsize=16)
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [X_demo.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('demographic_feature_importance.png')
        plt.close()
        print("\nDemographic feature importance saved as 'demographic_feature_importance.png'")

# Main execution
if __name__ == "__main__":
    # Load and explore data
    filepath = "data.csv"  # Update with your actual file path
    df = load_and_explore_data(filepath)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    X = df_processed.drop('fasting_ability', axis=1)
    y = df_processed['fasting_ability']
    
    # Analyze feature importance
    model, important_indices = analyze_feature_importance(X, y)
    
    # Train and evaluate the model
    best_model, X_test, y_test = train_and_evaluate_model(X, y)
    
    # Medical factors analysis
    analyze_medical_factors(df_processed, best_model)
    
    # Demographic factors analysis
    analyze_demographic_factors(df_processed, best_model)
    
    print("\nAnalysis complete! All visualizations have been saved.")