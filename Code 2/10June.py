import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, roc_curve, 
                            average_precision_score, f1_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create directories for organized output
def create_directories():
    """Create directories for organized visualization output"""
    directories = ['edaimages', 'visualizationquestions']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_and_explore_data(filepath):
    """Load and perform basic EDA with optimized visualizations"""
    print("="*60)
    print("DATA LOADING AND EXPLORATION")
    print("="*60)
    
    df = pd.read_csv(filepath, header=0)
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    missing_summary = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
    if not missing_summary.empty:
        print(missing_summary)
    else:
        print("No missing values found")
    
    # Target variable distribution
    print("\nTarget variable distribution:")
    target_dist = df['fasting_ability'].value_counts(normalize=True) * 100
    print(target_dist)
    
    # EDA Visualizations
    create_eda_visualizations(df)
    
    return df

def create_eda_visualizations(df):
    """Create EDA visualizations and save to edaimages folder"""
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    target_counts = df['fasting_ability'].value_counts()
    colors = ['#ff7f7f', '#7fbf7f']
    plt.pie(target_counts.values, labels=['Cannot Fast', 'Can Fast'], autopct='%1.1f%%', 
            startangle=90, colors=colors)
    plt.title('Distribution of Fasting Ability', fontsize=14, fontweight='bold')
    plt.savefig('edaimages/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Missing Values Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('edaimages/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Numerical Features Distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'fasting_ability' in numerical_cols:
        numerical_cols.remove('fasting_ability')
    
    if numerical_cols:
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(n_rows, n_cols, i+1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {col}', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('edaimages/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Categorical Features Distribution
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4*n_rows))
        for i, col in enumerate(categorical_cols):
            plt.subplot(n_rows, n_cols, i+1)
            value_counts = df[col].value_counts()
            plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            plt.title(f'Distribution of {col}', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig('edaimages/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Correlation Matrix
    numerical_cols_with_target = numerical_cols + ['fasting_ability']
    if len(numerical_cols_with_target) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numerical_cols_with_target].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('edaimages/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("EDA visualizations saved to 'edaimages' folder")

def preprocess_data(df):
    """Optimized preprocessing with faster operations"""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    df_processed = df.copy()
    
    # Drop ID column if exists
    if 'id' in df_processed.columns:
        df_processed.drop('id', axis=1, inplace=True)
    
    # Efficient cleaning functions
    def clean_gender(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).lower()
        if x_str in ['male', 'm']:
            return 0
        elif x_str in ['female', 'f']:
            return 1
        return np.nan
    
    def clean_boolean(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).lower()
        if x_str in ['true', 't', 'yes', 'y', '1']:
            return 1
        elif x_str in ['false', 'f', 'no', 'n', '0']:
            return 0
        return np.nan
    
    # Apply cleaning
    if 'gender_identity' in df_processed.columns:
        df_processed['gender_identity'] = df_processed['gender_identity'].apply(clean_gender)
    
    # Clean boolean columns
    boolean_columns = ['liver_function_impaired']
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(clean_boolean)
    
    # Encode categorical variables
    if 'ethnic_group' in df_processed.columns:
        encoder = LabelEncoder()
        df_processed['ethnic_group'] = encoder.fit_transform(df_processed['ethnic_group'].astype(str))
    
    # Feature engineering
    if 'body_mass_index' in df_processed.columns:
        df_processed['bmi_category'] = pd.cut(
            df_processed['body_mass_index'], 
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    if 'age' in df_processed.columns:
        df_processed['age_group'] = pd.cut(
            df_processed['age'],
            bins=[0, 18, 40, 65, 120],
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    # Efficient imputation
    if df_processed.isnull().any().any():
        print("Applying KNN imputation...")
        imputer = KNNImputer(n_neighbors=5)
        df_processed = pd.DataFrame(
            imputer.fit_transform(df_processed),
            columns=df_processed.columns
        )
        print("Imputation completed")
    
    return df_processed

def analyze_research_question_1(df, model):
    """Medical conditions analysis - Research Question 1"""
    print("\n" + "="*60)
    print("RESEARCH QUESTION 1: Medical Conditions Analysis")
    print("="*60)
    
    # Medical columns mapping
    medical_columns = {
        'liver_function_impaired': 'Liver Function Impaired',
        # Add more medical columns as they appear in your data
    }
    
    # Filter existing columns
    available_medical = {k: v for k, v in medical_columns.items() if k in df.columns}
    
    if not available_medical:
        print("No medical columns found in dataset")
        return
    
    # 1. Medical Conditions vs Fasting Ability
    plt.figure(figsize=(12, 6))
    
    medical_stats = []
    for i, (col, label) in enumerate(available_medical.items()):
        # Calculate fasting rates by condition
        condition_stats = df.groupby(col)['fasting_ability'].agg(['mean', 'count']).reset_index()
        condition_stats['percentage'] = condition_stats['mean'] * 100
        condition_stats['condition'] = label
        condition_stats['status'] = condition_stats[col].map({0: 'Normal', 1: 'Impaired'})
        medical_stats.append(condition_stats)
    
    # Combine all medical statistics
    if medical_stats:
        combined_stats = pd.concat(medical_stats, ignore_index=True)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=combined_stats, x='condition', y='percentage', hue='status')
        plt.title('Fasting Ability by Medical Conditions', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage Able to Fast (%)')
        plt.xlabel('Medical Condition')
        plt.legend(title='Condition Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizationquestions/medical_conditions_fasting.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Detailed Medical Analysis
    for col, label in available_medical.items():
        plt.figure(figsize=(10, 6))
        
        # Cross-tabulation
        crosstab = pd.crosstab(df[col], df['fasting_ability'], normalize='index') * 100
        
        # Stacked bar chart
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), 
                     color=['#ff7f7f', '#7fbf7f'])
        plt.title(f'Fasting Ability Distribution by {label}', fontsize=14, fontweight='bold')
        plt.xlabel('Condition Status (0=Normal, 1=Impaired)')
        plt.ylabel('Percentage (%)')
        plt.legend(['Cannot Fast', 'Can Fast'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'visualizationquestions/medical_{col}_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical summary
        print(f"\n{label} Analysis:")
        fasting_by_condition = df.groupby(col)['fasting_ability'].agg(['mean', 'count'])
        print(fasting_by_condition)
    
    print("Research Question 1 visualizations saved to 'visualizationquestions' folder")

def analyze_research_question_2(df, model):
    """Demographic reliability analysis - Research Question 2"""
    print("\n" + "="*60)
    print("RESEARCH QUESTION 2: Demographic Reliability Analysis")
    print("="*60)
    
    # Demographic columns
    demographic_columns = {
        'age': 'Age',
        'gender_identity': 'Gender',
        'body_mass_index': 'BMI',
        'bmi_category': 'BMI Category',
        'age_group': 'Age Group'
    }
    
    # Filter existing columns
    available_demographics = {k: v for k, v in demographic_columns.items() if k in df.columns}
    
    if not available_demographics:
        print("No demographic columns found")
        return
    
    # 1. Demographic Distribution by Fasting Ability
    n_demo = len(available_demographics)
    n_cols = min(3, n_demo)
    n_rows = (n_demo + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5*n_rows))
    
    for i, (col, label) in enumerate(available_demographics.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        if df[col].nunique() <= 10:  # Categorical
            demo_stats = df.groupby(col)['fasting_ability'].mean() * 100
            plt.bar(demo_stats.index, demo_stats.values, color='lightblue', alpha=0.7)
            plt.title(f'Fasting Ability by {label}')
            plt.ylabel('% Able to Fast')
        else:  # Continuous
            fasting_yes = df[df['fasting_ability'] == 1][col]
            fasting_no = df[df['fasting_ability'] == 0][col]
            
            plt.hist(fasting_no, alpha=0.5, label='Cannot Fast', bins=20, color='red')
            plt.hist(fasting_yes, alpha=0.5, label='Can Fast', bins=20, color='green')
            plt.title(f'Distribution of {label} by Fasting Ability')
            plt.xlabel(label)
            plt.ylabel('Frequency')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizationquestions/demographic_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Demographic Correlation Analysis
    demo_cols = list(available_demographics.keys()) + ['fasting_ability']
    demo_corr = df[demo_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(demo_corr, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Demographic Factors Correlation with Fasting Ability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizationquestions/demographic_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Build demographic-only model for reliability assessment
    X_demo = df[list(available_demographics.keys())]
    y = df['fasting_ability']
    
    # Quick train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_demo, y, test_size=0.2, random_state=42)
    
    # Simple model
    demo_model = RandomForestClassifier(n_estimators=50, random_state=42)
    demo_model.fit(X_train, y_train)
    
    # Predictions and accuracy
    y_pred = demo_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    importances = demo_model.feature_importances_
    feature_names = list(available_demographics.keys())
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.title('Demographic Feature Importance for Fasting Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('visualizationquestions/demographic_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reliability assessment
    print(f"\nDemographic Model Accuracy: {accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Reliability conclusion
    reliability_level = "HIGH" if accuracy >= 0.8 else "MODERATE" if accuracy >= 0.7 else "LOW"
    print(f"\nReliability Assessment: {reliability_level}")
    print(f"Demographic factors show {reliability_level.lower()} reliability for predicting fasting ability")
    
    print("Research Question 2 visualizations saved to 'visualizationquestions' folder")

def quick_model_training(X, y):
    """Optimized model training with reduced hyperparameter search"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Quick train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check class imbalance
    class_counts = np.bincount(y)
    imbalance_ratio = min(class_counts) / max(class_counts)
    
    # Apply SMOTE if needed
    if imbalance_ratio < 0.4:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Applied SMOTE for class balance")
    
    # Simplified hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Grid search with reduced CV
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Best Model Accuracy: {accuracy:.4f}")
    print(f"Best Model F1 Score: {f1:.4f}")
    
    return best_model

def main():
    """Main execution function"""
    print("Starting Ramadan Fasting Analysis...")
    
    # Create output directories
    create_directories()
    
    # Load and explore data
    df = load_and_explore_data('data.csv')
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    X = df_processed.drop('fasting_ability', axis=1)
    y = df_processed['fasting_ability']
    
    # Quick model training
    model = quick_model_training(X, y)
    
    # Answer research questions
    analyze_research_question_1(df_processed, model)
    analyze_research_question_2(df_processed, model)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("EDA images saved in: 'edaimages/' folder")
    print("Research question visualizations saved in: 'visualizationquestions/' folder")

if __name__ == "__main__":
    main()