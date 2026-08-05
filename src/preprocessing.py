# ============================================================
# preprocessing.py
# UMKM AI Business Resilience Prediction System
# Module: Data Preprocessing & Feature Engineering
# Author: AI Engineer
# Version: 1.0.0
# ============================================================

"""
Preprocessing module for UMKM Business Resilience Prediction System.

Handles:
    - Missing value detection and imputation
    - Duplicate detection and removal
    - Outlier detection (IQR method)
    - Data type validation
    - Feature engineering (construct scores)
    - Label encoding and one-hot encoding
    - Feature scaling (StandardScaler / MinMaxScaler)
    - Train-test split
"""

import warnings
import logging
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTRUCT COLUMN MAPPINGS
# ─────────────────────────────────────────────

CONSTRUCT_COLUMNS: Dict[str, List[str]] = {
    "DigitalCapabilityScore": ["DC1", "DC2", "DC3", "DC4", "DC5"],
    "InnovationCapabilityScore": ["IC1", "IC2", "IC3", "IC4", "IC5"],
    "EntrepreneurialOrientationScore": ["EO1", "EO2", "EO3", "EO4", "EO5"],
    "OrganizationalAgilityScore": ["OA1", "OA2", "OA3", "OA4", "OA5"],
    "ResourceAccessScore": ["RA1", "RA2", "RA3", "RA4", "RA5"],
    "EnvironmentalDynamismScore": ["ED1", "ED2", "ED3", "ED4", "ED5"],
    "BusinessResilienceScore": ["BR1", "BR2", "BR3", "BR4", "BR5", "BR6", "BR7"],
}

CATEGORICAL_COLUMNS: List[str] = [
    "Province", "City", "Business_Sector",
    "Owner_Gender", "Education", "Legal_Status"
]

NUMERIC_PROFILE_COLUMNS: List[str] = [
    "Business_Age", "Number_of_Employees",
    "Annual_Revenue", "Digital_Sales_Percentage", "Owner_Age"
]

FEATURE_COLUMNS: List[str] = [
    "DigitalCapabilityScore",
    "InnovationCapabilityScore",
    "EntrepreneurialOrientationScore",
    "OrganizationalAgilityScore",
    "ResourceAccessScore",
    "EnvironmentalDynamismScore",
]

TARGET_COLUMN: str = "BusinessResilienceCategory"


# ─────────────────────────────────────────────
# 1. BASIC DATA CHECKS
# ─────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing value counts and percentages per column.

    Args:
        df: Input DataFrame.

    Returns:
        Summary DataFrame with columns [Missing_Count, Missing_Pct].
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Pct": missing_pct
    })
    summary = summary[summary["Missing_Count"] > 0]
    if summary.empty:
        logger.info("✅ No missing values detected.")
    else:
        logger.warning(f"⚠️ Missing values found:\n{summary}")
    return summary


def check_duplicates(df: pd.DataFrame, id_col: str = "ID_UMKM") -> pd.DataFrame:
    """
    Detect and remove duplicate rows based on ID column.

    Args:
        df: Input DataFrame.
        id_col: Column to check for duplicates.

    Returns:
        Cleaned DataFrame with duplicates removed.
    """
    n_dup = df.duplicated(subset=[id_col]).sum()
    if n_dup > 0:
        logger.warning(f"⚠️ Found {n_dup} duplicate ID(s). Removing...")
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    else:
        logger.info(f"✅ No duplicate IDs found in '{id_col}'.")
    return df


def validate_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    """
    Validate and report data types for each column.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary mapping column names to their data types.
    """
    dtype_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
    logger.info(f"✅ Data types validated. Total columns: {len(dtype_map)}")
    return dtype_map


# ─────────────────────────────────────────────
# 2. OUTLIER DETECTION
# ─────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Args:
        df: Input DataFrame.
        columns: List of numeric columns to check.

    Returns:
        DataFrame summarising outlier counts per column.
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        records.append({
            "Column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": n_out,
        })
    summary = pd.DataFrame(records)
    logger.info(f"✅ Outlier detection complete for {len(columns)} columns.")
    return summary


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers using Z-Score method.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to evaluate.
        threshold: Z-score threshold (default 3.0).

    Returns:
        Boolean mask DataFrame; True indicates an outlier.
    """
    numeric_df = df[columns].select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df, nan_policy="omit"))
    mask = pd.DataFrame(z_scores > threshold, columns=numeric_df.columns)
    total_outliers = mask.sum().sum()
    logger.info(f"✅ Z-Score outlier check: {total_outliers} outlier cells detected (threshold={threshold}).")
    return mask


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_construct_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite latent variable scores by averaging Likert items.

    Each construct score is the mean of its indicator items.
    Scores range from 1.0 to 5.0.

    Args:
        df: Input DataFrame containing Likert item columns.

    Returns:
        DataFrame with additional composite score columns appended.
    """
    df = df.copy()
    for score_col, indicators in CONSTRUCT_COLUMNS.items():
        available = [c for c in indicators if c in df.columns]
        if not available:
            logger.warning(f"⚠️ No indicators found for {score_col}. Skipping.")
            continue
        df[score_col] = df[available].mean(axis=1).round(3)
        logger.info(f"✅ Computed {score_col} from {available}")
    return df


def create_resilience_label(
    df: pd.DataFrame,
    score_col: str = "BusinessResilienceScore",
    label_col: str = "BusinessResilienceCategory",
    low_threshold: float = 2.5,
    high_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Create a three-class Business Resilience target label.

    Rules:
        - Low    : score <= low_threshold
        - Medium : low_threshold < score <= high_threshold
        - High   : score > high_threshold

    Args:
        df: DataFrame with BusinessResilienceScore column.
        score_col: Name of the numeric resilience score column.
        label_col: Name for the new label column.
        low_threshold: Upper boundary for the 'Low' class.
        high_threshold: Upper boundary for the 'Medium' class.

    Returns:
        DataFrame with appended label column.
    """
    df = df.copy()

    def _categorize(score: float) -> str:
        if score <= low_threshold:
            return "Low"
        elif score <= high_threshold:
            return "Medium"
        else:
            return "High"

    df[label_col] = df[score_col].apply(_categorize)
    dist = df[label_col].value_counts()
    logger.info(f"✅ Label distribution:\n{dist.to_string()}")
    return df


# ─────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────

def encode_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the string target column into integers using LabelEncoder.

    Mapping: High → 2, Low → 0, Medium → 1 (alphabetical).

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (DataFrame with encoded target, fitted LabelEncoder).
    """
    df = df.copy()
    le = LabelEncoder()
    df[target_col + "_Encoded"] = le.fit_transform(df[target_col])
    logger.info(f"✅ Target encoded. Classes: {list(le.classes_)}")
    return df, le


def encode_categoricals(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode categorical (string) columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode; defaults to CATEGORICAL_COLUMNS.
        drop_first: Whether to drop the first dummy column (reduces multicollinearity).

    Returns:
        DataFrame with categorical columns replaced by dummies.
    """
    if columns is None:
        columns = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    logger.info(f"✅ One-hot encoded {len(columns)} categorical columns.")
    return df


# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Scale numeric features using StandardScaler or MinMaxScaler.

    Fit only on training data to prevent data leakage.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.
        method: Scaling method ('standard' or 'minmax').

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }
    if method not in scaler_map:
        raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard' or 'minmax'.")

    scaler = scaler_map[method]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info(f"✅ Features scaled using {scaler.__class__.__name__}.")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 6. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing subsets.

    Args:
        df: Full processed DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        test_size: Proportion of test data (default 0.20 → 80/20 split).
        random_state: Reproducibility seed.
        stratify: Whether to stratify split by target distribution.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[feature_cols]
    y = df[target_col]

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )
    logger.info(
        f"✅ Train-test split complete | "
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Test ratio: {test_size}"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    scale_method: str = "standard",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline end-to-end.

    Steps:
        1. Check missing values
        2. Remove duplicates
        3. Validate dtypes
        4. Detect outliers
        5. Compute construct scores
        6. Create resilience labels
        7. Encode target
        8. Split data
        9. Scale features

    Args:
        df: Raw loaded DataFrame.
        feature_cols: Features to use for modelling.
        target_col: Target column name.
        scale_method: 'standard' or 'minmax'.
        test_size: Train-test split ratio.
        random_state: Random seed.

    Returns:
        Dictionary with all preprocessing artefacts.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    logger.info("=" * 60)
    logger.info("🔄 Starting Preprocessing Pipeline")
    logger.info("=" * 60)

    # Step 1-3: Quality checks
    missing_summary = check_missing_values(df)
    df = check_duplicates(df)
    dtype_map = validate_dtypes(df)

    # Step 4: Outlier detection
    likert_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["DC", "IC", "EO", "OA", "RA", "ED", "BR"]
    )]
    outlier_summary = detect_outliers_iqr(df, likert_cols + NUMERIC_PROFILE_COLUMNS)

    # Step 5: Feature engineering
    df = compute_construct_scores(df)

    # Step 6: Label creation
    df = create_resilience_label(df)

    # Step 7: Encode target
    df, label_encoder = encode_target(df, target_col)

    # Step 8: Train-test split (using string labels for stratification)
    X_train, X_test, y_train, y_test = split_data(
        df, feature_cols, target_col, test_size, random_state
    )

    # Step 9: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scale_method)

    logger.info("✅ Preprocessing Pipeline Complete.")

    return {
        "df_processed": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "missing_summary": missing_summary,
        "outlier_summary": outlier_summary,
        "dtype_map": dtype_map,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
