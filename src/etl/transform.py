import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict

from src.etl.load import save_data
from configs.paths import get_project_paths

log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "data_transformer.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

class DataTransformer:
    def __init__(self, data: pd.DataFrame, knn_neighbors: int = 5):
        self.data = data.copy()
        self.knn_neighbors = knn_neighbors
        self.metadata: Dict = {}

    def remove_duplicates(self):
        initial = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed = initial - len(self.data)
        logger.info(f"Removed {removed} duplicate rows.")
        return self

    def handle_missing(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        cat_cols = self.data.select_dtypes(include=['object']).columns

        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self.data[num_cols] = imputer.fit_transform(self.data[num_cols])

        for col in cat_cols:
            if self.data[col].isna().any():
                mode = self.data[col].mode()
                fill_value = mode[0] if not mode.empty else 'Unknown'
                self.data[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled NA in {col} with {fill_value}")
        return self

    def drop_irrelevant(self):
        drop_cols = ["user_id"] 
        existing = [c for c in drop_cols if c in self.data.columns]
        self.data.drop(columns=existing, inplace=True)
        logger.info(f"Dropped irrelevant columns: {existing}")
        return self

    def process_dates(self):
        if "record_date" in self.data.columns:
            self.data["record_date"] = pd.to_datetime(self.data["record_date"], errors="coerce")
            self.data["record_year"] = self.data["record_date"].dt.year
            self.data["record_month"] = self.data["record_date"].dt.month
            self.data["record_quarter"] = self.data["record_date"].dt.quarter
            self.data["record_is_month_end"] = self.data["record_date"].dt.is_month_end.astype(int)
            self.data.drop(columns=["record_date"], inplace=True)
            logger.info("Processed record_date into time features.")
        return self

    def create_features(self):
        if {"monthly_income_usd", "monthly_expenses_usd"}.issubset(self.data.columns):
            self.data["disposable_income"] = self.data["monthly_income_usd"] - self.data["monthly_expenses_usd"]

        if {"loan_amount_usd", "monthly_income_usd"}.issubset(self.data.columns):
            self.data["loan_to_income_ratio"] = (
                self.data["loan_amount_usd"] / (self.data["monthly_income_usd"] + 1)
            )

        if "credit_score" in self.data.columns:
            bins = [0, 579, 669, 739, 799, 850]
            labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
            self.data["credit_score_bucket"] = pd.cut(self.data["credit_score"], bins=bins, labels=labels)

        return self

    def handle_outliers(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            self.data[col] = np.clip(self.data[col], lower, upper)
        logger.info("Applied IQR clipping to outliers.")
        return self

    def encode_categoricals(self):
        cat_cols = self.data.select_dtypes(include='object').columns.tolist()

        binary_cols = [c for c in cat_cols if self.data[c].nunique() == 2]
        for col in binary_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            cat_cols.remove(col)

        if len(cat_cols) > 0:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ct = ColumnTransformer(transformers=[('ohe', ohe, cat_cols)], remainder='passthrough')
            transformed = ct.fit_transform(self.data)
            new_cols = ct.get_feature_names_out()
            self.data = pd.DataFrame(transformed, columns=new_cols, index=self.data.index)

        logger.info("Encoded categorical variables.")
        return self

    def transform(self) -> pd.DataFrame:
        paths = get_project_paths()
        processed_dir = paths["PROCESSED"]
        transformed_dir = paths["TRANSFORMED"]

        formats = {
            "csv": {"index": False},
            "parquet": {"index": False, "compression": "snappy"},
            "sqlite": {"if_exists": "replace"}
        }

        self.remove_duplicates()\
            .drop_irrelevant()\
            .handle_missing()\
            .process_dates()\
            .create_features()\
            .handle_outliers()

        save_data(self.data, transformed_dir, "transformed_financial_dataset", formats=formats)

        self.encode_categoricals()

        save_data(self.data, processed_dir, "processed_financial_dataset", formats=formats)

        return self.data
