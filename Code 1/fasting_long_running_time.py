import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold, StratifiedKFold
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
    
    # Visualize medical conditions correlation with fasting ability
    medical_columns = ['liver_function_impaired']
    medical_columns = [col for col in medical_columns if col in df_processed.columns]
    
    if medical_columns:
        print("\nAnalyzing medical conditions correlation with fasting ability...")
        plt.figure(figsize=(12, len(medical_columns) * 4))
        for i, col in enumerate(medical_columns, 1):
            plt.subplot(len(medical_columns), 1, i)
            sns.countplot(x=col, hue='fasting_ability', data=df_processed)
            plt.title(f'Fasting Ability by {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('medical_conditions_vs_fasting.png')
        plt.close()
    
    # Demographic reliability analysis
    demographic_columns = ['gender_identity', 'body_mass_index', 'age']
    demographic_columns = [col for col in demographic_columns if col in df_processed.columns]
    
    if demographic_columns:
        print("\nAnalyzing demographic factors relation to fasting ability...")
        fig, axs = plt.subplots(len(demographic_columns), 1, figsize=(12, len(demographic_columns) * 5))
        
        for i, col in enumerate(demographic_columns):
            if df_processed[col].dtype in ['int64', 'float64']:
                if len(demographic_columns) > 1:
                    ax = axs[i]
                else:
                    ax = axs
                    
                # For numerical variables, use histograms
                sns.histplot(data=df_processed, x=col, hue='fasting_ability', multiple='dodge', bins=20, ax=ax)
                ax.set_title(f'Distribution of {col} by Fasting Ability')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('demographic_vs_fasting.png')
        plt.close()
    
    # Impute missing values with KNN
    print("\nImputing missing values with KNN...")
    
    # First, identify columns with missing values
    cols_with_missing = df_processed.columns[df_processed.isnull().any()].tolist()
    if cols_with_missing:
        print(f"Columns with missing values: {cols_with_missing}")
        
        # KNN imputation for all features
        imputer = KNNImputer(n_neighbors=5)
        df_processed_imputed = pd.DataFrame(
            imputer.fit_transform(df_processed),
            columns=df_processed.columns
        )
        
        # Check if imputation worked
        print("Missing values after imputation:", df_processed_imputed.isnull().sum().sum())
        
        return df_processed_imputed
    else:
        print("No missing values to impute")
        return df_processed

# Analyze feature importance
def analyze_feature_importance(X, y):
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature ranking:")
    for i, idx in enumerate(indices[:15]):  # Print top 15
        print(f"{i+1}. {X.columns[idx]} ({importances[idx]:.4f})")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance for Fasting Ability Prediction', fontsize=16)
    plt.bar(range(len(indices[:15])), importances[indices[:15]], align='center')
    plt.xticks(range(len(indices[:15])), [X.columns[i] for i in indices[:15]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Use permutation importance which is more reliable
    print("\nCalculating permutation importance...")
    # Split data for permutation importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_indices = perm_importance.importances_mean.argsort()[::-1]
    
    # Print permutation feature ranking
    print("\nPermutation Feature ranking:")
    for i, idx in enumerate(perm_indices[:15]):  # Print top 15
        print(f"{i+1}. {X.columns[idx]} ({perm_importance.importances_mean[idx]:.4f})")
    
    # Visualize permutation importance
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importance for Fasting Ability Prediction', fontsize=16)
    plt.bar(range(len(perm_indices[:15])), perm_importance.importances_mean[perm_indices[:15]], align='center')
    plt.xticks(range(len(perm_indices[:15])), [X.columns[i] for i in perm_indices[:15]], rotation=90)
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    plt.close()
    
    return model, indices, perm_indices

# Train and evaluate model
def train_and_evaluate_model(X, y):
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    # Check for class imbalance
    print("\nClass distribution:")
    class_counts = np.bincount(y)
    print(f"Class 0 (No Fasting): {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Class 1 (Fasting): {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    
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
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Random search with cross-validation
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rs = RandomizedSearchCV(
        rf, param_grid, n_iter=20, cv=cv, scoring='f1_weighted',
        random_state=42, n_jobs=-1
    )
    
    rs.fit(X_train, y_train)
    
    print(f"\nBest parameters: {rs.best_params_}")
    best_model = rs.best_estimator_
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='f1_weighted')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
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
    
    # Precision-Recall curve
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

# Medical analysis function
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
    
    # Create detailed analysis plots
    if medical_columns:
        plt.figure(figsize=(12, len(medical_columns) * 5))
        for i, col in enumerate(medical_columns):
            plt.subplot(len(medical_columns), 1, i + 1)
            
            if df[col].nunique() <= 10:  # For categorical/binary variables
                # Calculate percentages
                grouped = df.groupby(col)['fasting_ability'].mean().reset_index()
                grouped['percentage'] = grouped['fasting_ability'] * 100
                
                # Create bar plot
                ax = sns.barplot(x=col, y='percentage', data=grouped)
                
                # Add percentage labels on top of bars
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.1f}%", 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha = 'center', va = 'bottom')
                
                plt.title(f'Percentage of Patients Able to Fast by {col}')
                plt.ylabel('Percentage Able to Fast (%)')
            else:  # For continuous variables
                # Use a violin plot
                sns.violinplot(x='fasting_ability', y=col, data=df)
                plt.title(f'Distribution of {col} by Fasting Ability')
                plt.ylabel(col)
                plt.xlabel('Fasting Ability (0=No, 1=Yes)')
        
        plt.tight_layout()
        plt.savefig('medical_analysis.png')
        plt.close()
    
    # Subgroup analysis
    if 'liver_function_impaired' in df.columns:
        print("\nSubgroup analysis for patients with liver function impairment:")
        liver_patients = df[df['liver_function_impaired'] == 1]
        print(f"Number of patients with liver function impairment: {len(liver_patients)}")
        print(f"Percentage able to fast: {liver_patients['fasting_ability'].mean() * 100:.2f}%")
        
        # Compare with general population
        print(f"General population percentage able to fast: {df['fasting_ability'].mean() * 100:.2f}%")
        
        # Statistical comparison
        from scipy.stats import chi2_contingency
        
        contingency = pd.crosstab(df['liver_function_impaired'], df['fasting_ability'])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        print(f"Chi-square test: chi2 = {chi2:.4f}, p-value = {p:.4f}")
        print("Conclusion: " + ("Significant relationship" if p < 0.05 else "No significant relationship") + 
              " between liver function impairment and fasting ability")

# Demographic analysis function
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
    
    # Create detailed analysis charts
    for col in demographic_columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
            # Continuous variable - create binned analysis
            plt.figure(figsize=(12, 6))
            # Create bins for numerical data
            bins = pd.cut(df[col], bins=10)
            # Group by bins and calculate mean fasting ability
            grouped = df.groupby(bins)['fasting_ability'].agg(['mean', 'count']).reset_index()
            grouped['mean_percent'] = grouped['mean'] * 100
            
            # Plot as bar chart
            plt.bar(grouped[col].astype(str), grouped['mean_percent'], alpha=0.7)
            plt.title(f'Fasting Ability by {col} Ranges', fontsize=16)
            plt.xlabel(col)
            plt.ylabel('Percent Able to Fast (%)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'demographic_{col}_analysis.png')
            plt.close()
        elif df[col].nunique() <= 10:
            # Categorical variable
            plt.figure(figsize=(10, 6))
            # Group by category and calculate mean fasting ability
            grouped = df.groupby(col)['fasting_ability'].agg(['mean', 'count']).reset_index()
            grouped['mean_percent'] = grouped['mean'] * 100
            
            # Plot as bar chart
            plt.bar(grouped[col].astype(str), grouped['mean_percent'], alpha=0.7)
            plt.title(f'Fasting Ability by {col}', fontsize=16)
            plt.xlabel(col)
            plt.ylabel('Percent Able to Fast (%)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'demographic_{col}_analysis.png')
            plt.close()
    
    # Multivariate analysis - gender and BMI
    if 'gender_identity' in df.columns and 'bmi_category' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a pivot table
        pivot = pd.pivot_table(
            df, values='fasting_ability', 
            index='gender_identity', 
            columns='bmi_category', 
            aggfunc='mean'
        ) * 100
        
        # Plot as heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
        plt.title('Fasting Ability (%) by Gender and BMI Category', fontsize=16)
        plt.xlabel('BMI Category (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)')
        plt.ylabel('Gender (0=Male, 1=Female)')
        plt.tight_layout()
        plt.savefig('demographic_gender_bmi_analysis.png')
        plt.close()
    
    # Build a focused model on demographic variables only
    print("\nBuilding a model using only demographic variables...")
    X_demo = df[demographic_columns]
    y_demo = df['fasting_ability']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_demo, y_demo, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    model_demo = RandomForestClassifier(n_estimators=100, random_state=42)
    model_demo.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_demo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Demographic-only model accuracy: {accuracy:.4f}")
    print("\nClassification Report (Demographic-only model):")
    print(classification_report(y_test, y_pred))
    
    # Compare with full model
    print("\nConclusion: Reliability of demographic factors for prediction")
    print(f"Full model accuracy from previous analysis vs Demographic-only: {accuracy:.4f}")
    
    # Explanation of reliability
    if accuracy >= 0.75:
        print("The demographic factors appear to be HIGHLY RELIABLE predictors of fasting ability")
    elif accuracy >= 0.65:
        print("The demographic factors appear to be MODERATELY RELIABLE predictors of fasting ability")
    else:
        print("The demographic factors appear to be LESS RELIABLE predictors of fasting ability")

# Main function
def main():
    # Load and explore data
    df = load_and_explore_data('data.csv')
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    X = df_processed.drop('fasting_ability', axis=1)
    y = df_processed['fasting_ability']
    
    # Analyze feature importance
    model, indices, perm_indices = analyze_feature_importance(X, y)
    
    # Train and evaluate the model
    best_model, X_test, y_test = train_and_evaluate_model(X, y)
    
    # Answer Research Question 1: Relationship between medical conditions and fasting ability
    analyze_medical_factors(df_processed, best_model)
    
    # Answer Research Question 2: Reliability of demographic factors for prediction
    analyze_demographic_factors(df_processed, best_model)

if __name__ == "__main__":
    main()