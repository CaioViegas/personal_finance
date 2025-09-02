from pathlib import Path


def get_project_paths() -> dict:
    """
    Retorna um dicionário contendo os principais diretórios do projeto.
    """
    root = Path(__file__).resolve().parent.parent

    # Diretórios de dados
    data_dir = root / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    transformed_dir = data_dir / 'transformed'

    # Diretórios de código e notebooks
    src_dir = root / 'src'
    etl_dir = src_dir / 'etl'
    scripts_dir = root / 'scripts'
    notebooks_dir = root / 'notebooks'
    tests_dir = root / 'tests'

    # Diretórios de configuração, modelos e logs
    configs_dir = root / 'configs'
    models_dir = root / 'models'
    logs_dir = root / 'logs'

    return {
        'ROOT': root,
        'DATA': data_dir,
        'RAW': raw_dir,
        'PROCESSED': processed_dir,
        'TRANSFORMED': transformed_dir,
        'SRC': src_dir,
        'ETL': etl_dir,
        'SCRIPTS': scripts_dir,
        'NOTEBOOKS': notebooks_dir,
        'TESTS': tests_dir,
        'CONFIGS': configs_dir,
        'MODELS': models_dir,
        'LOGS': logs_dir,
    }