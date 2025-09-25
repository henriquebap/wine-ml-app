from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Encapsulates preprocessing steps: dedup, clip outliers, scaling."""

    def __init__(
        self,
        feature_columns: Optional[Iterable[str]] = None,
        target_column: str = "quality",
        outlier_clip_quantiles: Tuple[float, float] = (0.01, 0.99),
        scale_features: bool = True,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.target_column = target_column
        self.outlier_clip_quantiles = outlier_clip_quantiles
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        features = self._get_feature_columns(df)
        if self.scale_features:
            self.scaler = StandardScaler().fit(df[features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_proc = df.copy()
        df_proc = self._drop_duplicates(df_proc)
        df_proc = self._clip_outliers(df_proc)
        if self.scale_features and self.scaler is not None:
            features = self._get_feature_columns(df_proc)
            df_proc[features] = self.scaler.transform(df_proc[features])
        return df_proc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        if self.feature_columns is not None:
            return self.feature_columns
        return [c for c in df.columns if c != self.target_column]

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        q_low, q_high = self.outlier_clip_quantiles
        features = self._get_feature_columns(df)
        for col in features:
            low = df[col].quantile(q_low)
            high = df[col].quantile(q_high)
            df[col] = df[col].clip(lower=low, upper=high)
        return df


def save_processed(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


