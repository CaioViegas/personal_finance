import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.etl.load import save_data  
from configs.paths import get_project_paths 

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "finance_transformer_fixed.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_console.setFormatter(_formatter)

_file = logging.FileHandler(LOG_FILE)
_file.setLevel(logging.INFO)
_file.setFormatter(_formatter)

if not logger.hasHandlers():
    logger.addHandler(_console)
    logger.addHandler(_file)

class FinanceTransformer:
    def __init__(self, data: pd.DataFrame, *, knn_neighbors: int = 5, top_k_titles: int = 15, iqr_k: float = 1.5,) -> None:
        self.data = data.copy()
        self.knn_neighbors = knn_neighbors
        self.top_k_titles = top_k_titles
        self.iqr_k = iqr_k
        self.metadata: Dict[str, object] = {}

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _clip_non_negative(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].clip(lower=0)
        return df

    @staticmethod
    def _title_strip(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.strip()
            .str.replace(r"\\s+", " ", regex=True)
            .str.title()
        )

    def _normalize_missing_strings(self) -> None:
        self.data.replace(
            to_replace=r"^\\s*(nan|null|none|na|n/a)\\s*$",
            value=np.nan,
            regex=True,
            inplace=True,
        )

    def drop_duplicates(self) -> "FinanceTransformer":
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        removed = before - len(self.data)
        self.metadata["duplicates_removed"] = int(removed)
        logger.info("Linhas duplicadas removidas: %s", removed)
        return self

    def normalize_categoricals(self) -> "FinanceTransformer":
        cat_cols = [
            "gender",
            "education_level",
            "employment_status",
            "job_title",
            "has_loan",
            "loan_type",
            "region",
        ]
        for c in cat_cols:
            if c in self.data.columns:
                mask = self.data[c].notna()
                if mask.any():
                    self.data.loc[mask, c] = self._title_strip(self.data.loc[mask, c])

        if "has_loan" in self.data.columns:
            mapping = {
                "Yes": "Yes",
                "No": "No",
                "1": "Yes",
                "0": "No",
                "True": "Yes",
                "False": "No",
            }
            self.data.loc[self.data["has_loan"].notna(), "has_loan"] = (
                self.data.loc[self.data["has_loan"].notna(), "has_loan"].astype(str).str.title().map(mapping)
            )
        return self

    def convert_types(self) -> "FinanceTransformer":
        numeric_candidates = [
            "age",
            "monthly_income_usd",
            "monthly_expenses_usd",
            "savings_usd",
            "loan_amount_usd",
            "loan_term_months",
            "monthly_emi_usd",
            "loan_interest_rate_pct",
            "debt_to_income_ratio",
            "credit_score",
            "savings_to_income_ratio",
        ]
        self._coerce_numeric(self.data, numeric_candidates)

        if "record_date" in self.data.columns:
            self.data["record_date"] = pd.to_datetime(self.data["record_date"], errors="coerce")
        return self

    def enforce_business_bounds(self) -> "FinanceTransformer":
        non_neg = [
            "monthly_income_usd",
            "monthly_expenses_usd",
            "savings_usd",
            "loan_amount_usd",
            "loan_term_months",
            "monthly_emi_usd",
            "debt_to_income_ratio",
            "savings_to_income_ratio",
            "loan_interest_rate_pct",
        ]
        self._clip_non_negative(self.data, non_neg)

        if "credit_score" in self.data.columns:
            self.data["credit_score"] = self.data["credit_score"].clip(lower=300, upper=850)

        if "loan_interest_rate_pct" in self.data.columns:
            self.data["loan_interest_rate_pct"] = self.data["loan_interest_rate_pct"].clip(0, 30)
        return self

    def fill_categoricals(self) -> "FinanceTransformer":
        if "loan_type" in self.data.columns and "has_loan" in self.data.columns:
            cond_no = self.data["has_loan"].eq("No") & self.data["loan_type"].isna()
            cond_yes = self.data["has_loan"].eq("Yes") & self.data["loan_type"].isna()
            self.data.loc[cond_no, "loan_type"] = "None"
            self.data.loc[cond_yes, "loan_type"] = "Unknown"

        for c in self.data.select_dtypes(include="object").columns:
            if self.data[c].isna().any():
                mode = self.data[c].dropna().mode()
                fill = mode.iloc[0] if not mode.empty else "Unknown"
                self.data[c] = self.data[c].fillna(fill)
                logger.info("Preenchidos nulos em '%s' com '%s'", c, fill)
        return self

    def impute_numbers(self) -> "FinanceTransformer":
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return self
        before_na = int(self.data[num_cols].isna().sum().sum())
        if before_na == 0:
            return self
        imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
        after_na = int(self.data[num_cols].isna().sum().sum())
        logger.info("KNN preencheu %s valores numéricos faltantes", before_na - after_na)
        return self

    def reduce_cardinality(self) -> "FinanceTransformer":
        col = "job_title"
        if col in self.data.columns:
            topk = (
                self.data[col]
                .value_counts(dropna=False)
                .nlargest(self.top_k_titles)
                .index
                .tolist()
            )
            self.data[col] = np.where(self.data[col].isin(topk), self.data[col], "Other")
            self.metadata["job_title_topk"] = topk
            logger.info("'job_title' reduzido para Top-%s + 'Other'", self.top_k_titles)
        return self

    def treat_outliers(self) -> "FinanceTransformer":
        cols = [
            "loan_term_months",
            "monthly_emi_usd",
            "debt_to_income_ratio",
            "loan_amount_usd",
            "monthly_expenses_usd",
            "monthly_income_usd",
            "savings_usd",
        ]
        for c in cols:
            if c not in self.data.columns:
                continue
            q1 = self.data[c].quantile(0.25)
            q3 = self.data[c].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lo = q1 - self.iqr_k * iqr
            hi = q3 + self.iqr_k * iqr
            before = self.data[c].copy()
            self.data[c] = self.data[c].clip(lower=lo, upper=hi)
            changed = int((before != self.data[c]).sum())
            if changed:
                logger.info("Outliers cap em '%s': %s valores ajustados", c, changed)
        return self

    def process_dates(self) -> "FinanceTransformer":
        if "record_date" not in self.data.columns:
            return self
        dt = self.data["record_date"]
        self.data["record_year"] = dt.dt.year
        self.data["record_month"] = dt.dt.month
        self.data["record_day"] = dt.dt.day
        self.data["record_weekday"] = dt.dt.weekday
        self.data["record_quarter"] = dt.dt.quarter
        self.data["record_is_month_end"] = dt.dt.is_month_end.astype(int)
        return self

    def create_features(self) -> "FinanceTransformer":
        d = self.data
        income = d.get("monthly_income_usd")
        expenses = d.get("monthly_expenses_usd")
        emi = d.get("monthly_emi_usd")
        savings = d.get("savings_usd")

        if income is not None and expenses is not None:
            self.data["net_income_usd"] = income - expenses
            self.data["expense_ratio"] = (expenses / income.replace(0, np.nan)).fillna(0)
        if income is not None and emi is not None:
            self.data["emi_to_income_ratio"] = (emi / income.replace(0, np.nan)).fillna(0)
        if income is not None and savings is not None:
            self.data["savings_rate_monthly"] = (savings / (income * 12).replace(0, np.nan)).fillna(0)

        if "debt_to_income_ratio" in self.data.columns and "emi_to_income_ratio" in self.data.columns:
            dti = self.data["debt_to_income_ratio"].copy()
            est = (self.data["emi_to_income_ratio"] * 100).round(2)
            use_est = dti.isna() | (dti == 0)
            self.data["debt_to_income_ratio_filled"] = np.where(use_est, est, dti)

        if "credit_score" in self.data.columns:
            bins = [299, 579, 669, 739, 799, 850]
            labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
            self.data["credit_bucket"] = pd.cut(self.data["credit_score"], bins=bins, labels=labels)

        if "loan_term_months" in self.data.columns:
            self.data["loan_term_years"] = (self.data["loan_term_months"] / 12.0).round(2)

        if {"has_loan", "loan_amount_usd"}.issubset(self.data.columns):
            self.data["loan_active"] = (
                (self.data["has_loan"].eq("Yes")) & (self.data["loan_amount_usd"] > 0)
            ).astype(int)

        parts = []
        if "expense_ratio" in self.data.columns:
            parts.append(self.data["expense_ratio"].clip(0, 1))
        if "emi_to_income_ratio" in self.data.columns:
            parts.append(self.data["emi_to_income_ratio"].clip(0, 1))
        if "debt_to_income_ratio_filled" in self.data.columns:
            parts.append((self.data["debt_to_income_ratio_filled"] / 100).clip(0, 1))
        if parts:
            self.data["financial_stress_score"] = np.vstack(parts).mean(axis=0)
        return self

    def encode_for_model(self, target: Optional[pd.Series] = None) -> pd.DataFrame:
        df = self.data.copy()
        drop_cols = [c for c in ["user_id", "record_date"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        if "has_loan" in df.columns:
            df["has_loan"] = df["has_loan"].map({"Yes": 1, "No": 0}).astype("Int64")

        ordinal_maps = {
            "education_level": ["High School", "Bachelor", "Master", "Phd"],
            "credit_bucket": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
        }
        for col, order in ordinal_maps.items():
            if col in df.columns:
                ord_enc = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
                df[col] = ord_enc.fit_transform(df[[col]])

        low_card = [c for c in ["gender", "employment_status", "region"] if c in df.columns]
        if low_card:
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            ohe_arr = ohe.fit_transform(df[low_card])
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(low_card), index=df.index)
            df = pd.concat([df.drop(columns=low_card), ohe_df], axis=1)

        high_card = [c for c in ["job_title", "loan_type"] if c in df.columns]
        if high_card and target is not None:
            te = TargetEncoder(cols=high_card)
            df[high_card] = te.fit_transform(df[high_card], target)

        return df

    def transform(self) -> pd.DataFrame:
        paths = get_project_paths()
        processed_dir = paths["PROCESSED"]
        transformed_dir = paths["TRANSFORMED"]

        formats = {
            "csv": {"index": False, "compression": None},
            "parquet": {"index": False, "compression": "snappy"},
            "sqlite": {"if_exists": "replace"},
        }

        self._normalize_missing_strings()

        (
            self.drop_duplicates()
            .normalize_categoricals()
            .convert_types()
            .enforce_business_bounds()
            .fill_categoricals()
            .impute_numbers()
            .reduce_cardinality()
            .treat_outliers()
            .process_dates()
            .create_features()
        )

        save_data(self.data, transformed_dir, "transformed_finance_data", formats=formats)

        processed = self.encode_for_model()
        save_data(processed, processed_dir, "processed_finance_data", formats=formats)

        logger.info("Transformação concluída. Shape final (processed): %s", processed.shape)
        return processed


__all__ = ["FinanceTransformer"]
