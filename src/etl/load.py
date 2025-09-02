import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Union

paths = Path(__file__).resolve().parent.parent.parent / "logs"
paths.mkdir(parents=True, exist_ok=True)
log_file = paths / "save_data.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def save_data(df: pd.DataFrame, save_dir: Union[str, Path], base_filename: str = "data", formats: Optional[Dict[str, Dict]] = None, sqlite_table: str = "data_table", timestamp: bool = False) -> Dict[str, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        from datetime import datetime
        base_filename = f"{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    default_formats = {
        "csv": {"index": False, "compression": "gzip"},
        "parquet": {"index": False, "compression": "snappy"},
        "sqlite": {"if_exists": "replace"}
    }
    formats = formats or default_formats
    
    saved_files = {}
    
    try:
        if "csv" in formats:
            compression = formats["csv"].get("compression", None)
            ext = ".csv.gz" if compression in ["gzip", "gz"] else ".csv"
            csv_path = save_dir / f"{base_filename}{ext}"
            df.to_csv(csv_path, **formats["csv"])
            saved_files["csv"] = csv_path
            logger.info(f"Saved to CSV ({'compressed' if compression else 'uncompressed'}): {csv_path}")
        
        if "parquet" in formats:
            parquet_path = save_dir / f"{base_filename}.parquet"
            df.to_parquet(parquet_path, **formats["parquet"])
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved to Parquet: {parquet_path}")
        
        if "sqlite" in formats:
            sqlite_path = save_dir / f"{base_filename}.db"
            conn = sqlite3.connect(sqlite_path)
            df.to_sql(sqlite_table, conn, **formats["sqlite"])
            conn.close()
            saved_files["sqlite"] = sqlite_path
            logger.info(f"Saved to SQLite (table={sqlite_table}): {sqlite_path}")
            
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}", exc_info=True)
        raise
    
    return saved_files