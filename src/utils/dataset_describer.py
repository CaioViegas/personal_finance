import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from scipy.stats import zscore
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_outliers(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, rare_threshold: float = 0.05, z_thresh: float = 3, error_cols: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Detects outliers in a given Pandas DataFrame for specified numeric columns and returns a summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze for outliers.
    numeric_cols : Optional[List[str]], optional
        A list of column names to check for outliers. If None, all numeric columns are considered.
    rare_threshold : float, optional
        The threshold for classifying an outlier as rare but possible. Defaults to 0.05.
    z_thresh : float, optional
        The Z-score threshold for detecting outliers. Defaults to 3.
    error_cols : Optional[Dict[str, float]], optional
        A dictionary specifying minimum acceptable values for certain columns, used to identify obvious errors. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for each checked column, indicating the number of outliers, percentage of outliers,
        minimum and maximum values, classification of the column, and recommended action.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    if error_cols is None:
        error_cols = {"use_count": 0, "average_basket_size": 0, "membership_fee": 0}  

    results = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            logger.warning(f"Coluna '{col}' vazia após remoção de NaN.")
            continue
        
        z_scores = np.abs(zscore(series))
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        outlier_mask = ((z_scores > z_thresh) | (series < bounds[0]) | (series > bounds[1]))
        num_outliers = outlier_mask.sum()
        pct_outliers = num_outliers / len(series)

        if col in error_cols and series.min() < error_cols[col]:
            classification, action = "Obvious error", "Fix or remove"

        elif pct_outliers == 0:
            classification, action = "No outliers", "No action needed"

        elif pct_outliers < rare_threshold:
            classification, action = "Rare but possible", "Keep or segment"
            
        else:
            classification, action = "May harm model", "Transform/use robust model"

        results.append({
            "column": col,
            "outliers": num_outliers,
            "percentage": round(pct_outliers * 100, 2),
            "min_value": series.min(),
            "max_value": series.max(),
            "classification": classification,
            "action": action
        })
    
    return pd.DataFrame(results)


def describe_dataset(df: pd.DataFrame, name: str = "dataset", display: bool = True, return_results: bool = True, save_to_file: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
    results = {}
    output_lines = []

    def _log(text):
        if display:
            print(text)
        output_lines.append(text)

    _log(f"\n{'='*40}")
    _log(f"EDA FOR DATASET: {name.upper()}")
    _log(f"{'='*40}")

    shape_info = pd.DataFrame({
        "Metric": ["Rows", "Columns"],
        "Count": [df.shape[0], df.shape[1]]
    })
    results["shape"] = shape_info
    _log("\n[1] SHAPE:")
    _log(shape_info.to_string(index=False))

    dtypes = df.dtypes.value_counts().reset_index()
    dtypes.columns = ["Type", "Count"]
    results["dtypes"] = dtypes
    _log("\n[2] DATA TYPES:")
    _log(dtypes.to_string(index=False))

    nulls = df.isnull().sum()
    missing = nulls[nulls > 0].sort_values(ascending=False).reset_index()
    missing.columns = ["Column", "Missing Values"]
    results["missing"] = missing
    _log("\n[3] MISSING VALUES:")
    if missing.empty:
        _log("No missing values found")
    else:
        _log(missing.to_string(index=False))

    if df.select_dtypes(include=np.number).shape[1] > 0:
        num_stats = df.describe(include=[np.number]).T
        results["numeric_stats"] = num_stats
        _log("\n[4] NUMERIC STATISTICS:")
        _log(num_stats.to_string(float_format="%.2f"))

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        cat_stats = pd.DataFrame({
            "Column": cat_cols,
            "Unique Values": [df[col].nunique() for col in cat_cols],
            "Most Common": [df[col].mode()[0] if not df[col].mode().empty else None for col in cat_cols],
            "Freq of Most Common": [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in cat_cols],
        })
        cat_stats["Cardinality Ratio"] = cat_stats["Unique Values"] / df.shape[0]
        results["categorical"] = cat_stats
        _log("\n[5] CATEGORICAL STATISTICS:")
        _log(cat_stats.to_string(index=False))

        high_card_cols = cat_stats[cat_stats["Cardinality Ratio"] > 0.5]["Column"].tolist()
        if high_card_cols:
            _log(f"\n[!] Warning: High-cardinality categorical columns: {', '.join(high_card_cols)}")

        cat_distributions = {
            col: df[col].value_counts(normalize=True).head(3).to_dict()
            for col in cat_cols
        }
        results["top_categories"] = pd.DataFrame.from_dict(cat_distributions, orient='index').fillna(0)
        _log("\n[6] TOP 3 CATEGORIES PER COLUMN (proportions):")
        _log(results["top_categories"].to_string(float_format="%.2f"))

    if df.select_dtypes(include=np.number).shape[1] > 0:
        outliers = detect_outliers(df)
        results["outliers"] = outliers

        if display:
            print("\n[6] OUTLIER ANALYSIS:")
            print(outliers.to_string(index=False))

        if save_to_file:
            output_lines.append("\n[6] OUTLIER ANALYSIS:")
            output_lines.append(outliers.to_string(index=False))

    if df.select_dtypes(include=np.number).shape[1] > 1:
        corr = df.select_dtypes(include=np.number).corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong_corrs = upper_tri.stack().reset_index()
        strong_corrs.columns = ["Var1", "Var2", "Correlation"]
        strong_corrs = strong_corrs[strong_corrs["Correlation"] >= 0.8]
        if not strong_corrs.empty:
            results["strong_correlations"] = strong_corrs
            _log("\n[8] STRONG CORRELATIONS (>= 0.8):")
            _log(strong_corrs.to_string(index=False))

    issues = []
    constant_cols = df.columns[df.nunique(dropna=False) == 1].tolist()
    if constant_cols:
        issues.append({"Issue": "Constant Columns", "Details": ", ".join(constant_cols)})

    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append({"Issue": "Duplicate Rows", "Count": duplicate_rows})

    if issues:
        issues_df = pd.DataFrame(issues)
        results["data_issues"] = issues_df
        _log("\n[9] DATA ISSUES:")
        _log(issues_df.to_string(index=False))

    _log(f"\n{'='*40}\n")

    if save_to_file:
        paths = get_project_paths()
        log_dir = paths['LOGS']
        out_path = Path(log_dir) / f"{name}_eda_summary.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)  

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        print(f"[i] Summary saved to file: {out_path.resolve()}")

    return results if return_results else None