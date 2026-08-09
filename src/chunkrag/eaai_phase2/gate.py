from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chunkrag.eaai_phase2.constants import CHUNKERS, NUMERIC_FEATURES
from chunkrag.eaai_phase2.features import validate_feature_row
from chunkrag.eaai_phase2.io import sha256_file


@dataclass(slots=True)
class ConstantGate:
    positive_probability: float

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(rows), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def feature_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    for row in rows:
        validate_feature_row(row)
    return pd.DataFrame(
        [{name: row[name] for name in (*NUMERIC_FEATURES, "chunker")} for row in rows],
        columns=[*NUMERIC_FEATURES, "chunker"],
    )


def build_logistic_gate() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        categories=[list(CHUNKERS)],
        handle_unknown="error",
        sparse_output=False,
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("chunker", categorical, ["chunker"]),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1_000,
        random_state=20_260_809,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def fit_gate(
    feature_rows: Sequence[dict[str, Any]],
    labels: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    if len(feature_rows) != len(labels) or not feature_rows:
        raise ValueError("Gate features and labels must be non-empty and aligned")
    label_array = np.asarray(labels, dtype=int)
    if not set(label_array.tolist()) <= {0, 1}:
        raise ValueError("Gate labels must be binary")
    frame = feature_frame(feature_rows)
    classes, counts = np.unique(label_array, return_counts=True)
    label_counts = {str(int(key)): int(value) for key, value in zip(classes, counts, strict=True)}
    if len(classes) == 1:
        model: Any = ConstantGate(float(classes[0]))
        model_type = "constant_fallback"
    else:
        model = build_logistic_gate()
        model.fit(frame, label_array)
        model_type = "l2_logistic_regression"
    metadata = {
        "model_type": model_type,
        "threshold": 0.5,
        "numeric_features": list(NUMERIC_FEATURES),
        "chunker_categories": list(CHUNKERS),
        "training_rows": len(feature_rows),
        "label_counts": label_counts,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }
    return model, metadata


def gate_probabilities(model: Any, feature_rows: Sequence[dict[str, Any]]) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(feature_frame(feature_rows)), dtype=float)
    if probabilities.shape != (len(feature_rows), 2):
        raise ValueError(f"Unexpected gate probability shape: {probabilities.shape}")
    return probabilities[:, 1]


def save_gate(model: Any, path: str | Path) -> str:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(output)
    return sha256_file(output)


def load_gate(path: str | Path, expected_sha256: str) -> Any:
    model_path = Path(path)
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_sha256:
        raise RuntimeError(f"Gate hash mismatch: {actual_hash} != {expected_sha256}")
    return joblib.load(model_path)
