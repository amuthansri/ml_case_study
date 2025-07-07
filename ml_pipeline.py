# imporing libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, average_precision_score, precision_recall_curve, auc
import joblib
import flask



def load_and_merge_datasets(excel_filename="DataScientist_CaseStudy_Dataset.xlsx"):
    # Get current working directory and build full path to the file
    current_path = os.getcwd()+"\\data"
    filepath = os.path.join(current_path, excel_filename)
    print("Current folder:", filepath)

    # Load the Excel file
    xls = pd.ExcelFile(filepath)

    # Parse sheets
    soc_dem_df = xls.parse("Soc_Dem")
    products_bal_df = xls.parse("Products_ActBalance")
    inflow_outflow_df = xls.parse("Inflow_Outflow")
    sales_revenues_df = xls.parse("Sales_Revenues")

    # Print basic info
    print(f"soc_dem_df counts: {soc_dem_df['Client'].count()}")
    print(f"products_bal_df counts: {products_bal_df['Client'].count()}")
    print(f"inflow_outflow_df counts: {inflow_outflow_df['Client'].count()}")
    print(f"sales_revenues_df counts: {sales_revenues_df['Client'].count()}")

    # Merge all datasets on 'Client'
    source_df = soc_dem_df.merge(products_bal_df, on='Client', how='left') \
                          .merge(inflow_outflow_df, on='Client', how='left') \
                          .merge(sales_revenues_df, on='Client', how='left')

    print(f"{len(source_df)} rows in the source data frame")

    return source_df



def prepare_target_dataset(source_df, target_column, test_size=0.3, random_state=42):
    """
    Prepares a dataset for training a model on the specified target column.
    Drops unrelated target columns, handles missing target rows, splits into
    X/y, and then performs train/test split.

    Parameters:
    - source_df (pd.DataFrame): The full merged dataset
    - target_column (str): One of 'Sale_MF', 'Sale_CC', or 'Sale_CL'
    - test_size (float): Fraction of data to use for test/validation
    - random_state (int): Random seed for reproducibility

    Returns:
    - X_train, X_temp, y_train, y_temp (tuple): Train and temp feature/target splits
    """

    valid_targets = ['Sale_MF', 'Sale_CC', 'Sale_CL', 'Revenue_CL', 'Revenue_CC', 'Revenue_MF']
    if target_column not in valid_targets:
        raise ValueError(f"Invalid target_column '{target_column}'. Must be one of {valid_targets}")

    # Drop other two target columns
    columns_to_drop = [col for col in valid_targets if col != target_column]
    target_df = source_df.drop(columns=columns_to_drop)

    # Drop rows where the selected target is missing
    target_df = target_df.dropna(subset=[target_column])

    # Split into features and target
    y = target_df[target_column]
    X = target_df.drop(columns=[target_column])

    # Perform train/temp split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f" X_train lenght is : {len(X_train)}")
    print(f" X_test lenght is : {len(X_test)}")
    print(f" y_train lenght is : {len(y_train)}")
    print(f" y_test lenght is : {len(y_test)}")

    return X_train, X_test, y_train, y_test


def preprocess_dataset(X, X_train):
    """
    Cleans, encodes, and preprocesses the input dataset X using X_train
    as reference for imputation.

    Parameters:
    - X (pd.DataFrame): Input dataset to clean (e.g., X_test)
    - X_train (pd.DataFrame): Reference dataset for imputation (e.g., training set)

    Returns:
    - pd.DataFrame: Cleaned and preprocessed dataset
    """

    X = X.copy()

    # 1. Remove 'Client' column
    if 'Client' in X.columns:
        X = X.drop(columns=['Client'], axis=1)

    # 2. Tenure: convert months to years
    if 'Tenure' in X.columns:
        X['Tenure'] = X['Tenure'] / 12

    # 3. Drop multicollinearity features if they exist
    high_corr_to_drop = [
        'VolumeCred_CA', 'VolumeDeb_CA', 'VolumeDeb_PaymentOrder',
        'TransactionsCred_CA', 'TransactionsDeb_CA', 'TransactionsDebCashless_Card',
        'ActBal_CL', 'VolumeDeb'
    ]
    cols_to_drop = [col for col in high_corr_to_drop if col in X.columns]
    X = X.drop(columns=cols_to_drop)

    # 4. Encode 'Sex' column (fill NaNs with mode, get_dummies)
    if 'Sex' in X.columns:
        mode_value = X['Sex'].mode()[0]
        X['Sex'] = X['Sex'].fillna(mode_value)

        sex_encoded = pd.get_dummies(X['Sex'], drop_first=True).astype(int)
        X = pd.concat([X.drop(columns=['Sex']), sex_encoded], axis=1)

    # 5. Impute missing values based on 80% rule
    for col in X.columns:
        if X[col].isnull().sum() > 0 and X[col].dtype != 'object':
            null_count = X[col].isnull().sum()
            total_count = len(X)
            null_percentage = (null_count / total_count) * 100

            if null_percentage > 80:
                X[col] = X[col].fillna(0)
            else:
                if col in X_train.columns:
                    X[col] = X[col].fillna(X_train[col].median())
                else:
                    X[col] = X[col].fillna(X[col].median())  # fallback if not in train

    # 6. Create missing value flags for selected features
    # for flag_col in ['ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC']:
    #     if flag_col in X.columns:
    #         X[f'{flag_col}_missing'] = X[flag_col].isnull().astype(int)

    return X



def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE oversampling to balance the training dataset.

    Parameters:
    - X_train (pd.DataFrame or np.array): Training features
    - y_train (pd.Series or np.array): Training labels
    - random_state (int): Seed for reproducibility

    Returns:
    - X_resampled: Resampled features
    - y_resampled: Resampled labels
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled



def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42, n_jobs=-1):
    """
    Trains a Random Forest Classifier on the provided training data.

    Parameters:
    - X_train (pd.DataFrame or np.array): Training features
    - y_train (pd.Series or np.array): Training labels
    - n_estimators (int): Number of trees in the forest (default: 100)
    - max_depth (int or None): Maximum depth of each tree (default: None = grow until all leaves are pure)
    - random_state (int): Seed for reproducibility
    - n_jobs (int): Number of CPU cores to use (-1 means use all cores)

    Returns:
    - rf: Trained RandomForestClassifier model
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, y_train)
    print(f" ******  training done **** ")
    return rf


def run_target_pipeline(target_column, prob_column_name, model_label='model', random_state=42):
    """
    Executes full ML pipeline for a given target column.
    
    Parameters:
    - target_column: str, one of 'Sale_MF', 'Sale_CC', 'Sale_CL'
    - prob_column_name: str, name of the output probability column (e.g., 'Prob_Purchase_MF')
    - model_label: optional name of the model variable returned
    - random_state: int, random seed
    
    Returns:
    - ranking_df: DataFrame with 'Client' and prediction probabilities
    - trained_model: fitted RandomForestClassifier model
    """

    # Load and prepare data
    source_df = load_and_merge_datasets()
    X_train, X_test, y_train, y_test = prepare_target_dataset(source_df, target_column=target_column)

    # Store Client IDs for final output
    ranking_df = pd.DataFrame({'Client': X_test['Client']})

    # Preprocess datasets
    X_train_processed = preprocess_dataset(X_train, X_train)
    X_test_processed = preprocess_dataset(X_test, X_train)

    # Handle imbalance
    X_resampled, y_resampled = apply_smote(X_train_processed, y_train, random_state=random_state)

    # Train model
    model = train_random_forest(X_resampled, y_resampled)

    # Predict probabilities
    y_scores = model.predict_proba(X_test_processed)[:, 1]

    # Assign to result
    ranking_df[prob_column_name] = y_scores

    return ranking_df, model
