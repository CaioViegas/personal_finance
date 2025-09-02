import sys
import kagglehub
import shutil
import pandas as pd
import logging
from pathlib import Path
from src.etl.load import save_data

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths
from src.utils.dataset_describer import describe_dataset

paths = get_project_paths()
log_dir = paths['LOGS']
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "dataset_download.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def download_dataset(slug: str, file_extension: str = ".csv") -> list[Path]:
    paths = get_project_paths()
    raw_dir = paths['RAW']
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        logger.info(f"Downloading dataset '{slug}'...")
        dataset_path = Path(kagglehub.dataset_download(slug))

        if not dataset_path.exists():
            raise FileNotFoundError(f"Kaggle download failed: {dataset_path} not found.")
        
        for file in dataset_path.glob(f"*{file_extension}"):
            dest_file = raw_dir / file.name

            shutil.copy(file, dest_file)
            saved_files.append(dest_file)
            logger.info(f"File saved: {dest_file}")

            try:
                df = pd.read_csv(dest_file)
                if df.empty:
                    logger.warning(f"Empty DataFrame: {file.name}")
                    continue
                    
                describe_dataset(df)
                save_data(df, save_dir=raw_dir, base_filename=file.stem)

            except pd.errors.EmptyDataError as e:
                logger.error(f"Empty CSV: {file.name} - {e}")

            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Download failed for '{slug}': {e}", exc_info=True)

    finally:
        if 'dataset_path' in locals() and dataset_path.exists():
            shutil.rmtree(dataset_path, ignore_errors=True)
            logger.info(f"Cleaned up temp dir: {dataset_path}")

    return saved_files

if __name__ == "__main__":
    download_dataset(
        "miadul/personal-finance-ml-dataset",
        file_extension=".csv"
    )