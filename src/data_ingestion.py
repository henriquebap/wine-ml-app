from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID: str = os.getenv("HF_DATASET_REPO", "henriquebap/wine-ml-dataset")
DEFAULT_FILENAME: str = os.getenv("HF_DATASET_FILENAME", "WineQT.csv")

FEATURE_COLUMNS: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def get_hf_token(env_var_name: str = "HF_TOKEN") -> Optional[str]:
    """Return the Hugging Face token from environment if available."""
    token = os.getenv(env_var_name)
    return token if token else None


def download_dataset_file(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    repo_type: str = "dataset",
    token_env: str = "HF_TOKEN",
) -> Path:
    """Download a file from a Hugging Face dataset repo and return local path.

    The file is cached by huggingface_hub.
    """
    token = get_hf_token(token_env)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
    )
    return Path(local_path)


def load_wine_dataframe(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    feature_columns: Optional[Iterable[str]] = None,
    token_env: str = "HF_TOKEN",
) -> pd.DataFrame:
    """Load wine dataset into a DataFrame, selecting expected columns and dropping NA."""
    csv_path = download_dataset_file(repo_id=repo_id, filename=filename, token_env=token_env)
    df = pd.read_csv(csv_path)

    expected_features = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    allowed_columns = set(expected_features + ["quality"])  # target column
    filtered_columns = [c for c in df.columns if c in allowed_columns]
    filtered_df = df[filtered_columns].dropna().reset_index(drop=True)
    return filtered_df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    """Save a DataFrame to CSV, ensuring parent directories exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_and_cache_raw(
    output_dir: Path = Path("data/raw"),
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Download and cache raw dataset under data/raw for local exploration."""
    df = load_wine_dataframe(repo_id=repo_id, filename=filename)
    output_path = output_dir / filename
    save_dataframe(df, output_path)
    return output_path


if __name__ == "__main__":
    df_main = load_wine_dataframe()
    print(f"Loaded dataframe with shape: {df_main.shape}")
    print(df_main.head())


