"""
train_automl.py  --  Steps 4 & 5: AutoML training with MLflow tracking
=======================================================================
MLOps pillars demonstrated here:
  * EXPERIMENT TRACKING   - every run's params/metrics/artifacts logged to MLflow
  * AUTOML                - FLAML searches algorithm families (LightGBM, XGBoost,
                            Random Forest, Extra Trees) and tunes hyperparameters
  * MODEL REGISTRY        - the best pipeline is registered for versioned serving
  * ARTIFACT LOGGING      - the fitted preprocessing+model pipeline is logged
  * REPRODUCIBILITY       - fixed seed; preprocessing baked into the pipeline so
                            training and serving apply identical transforms

Why a single sklearn Pipeline(preprocessor -> model)?
  The one-hot encoding + passthrough is stored together with the model, so the
  deployed app feeds RAW student rows and the pipeline handles encoding. This
  eliminates training/serving skew, a classic production ML failure mode.

Run:  python -m src.train_automl --time-budget 60
"""
import argparse
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, pass numeric features through untouched."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through
    )


def main(time_budget: int = 60):
    config.ensure_dirs()

    # --- Load the reproducible split ------------------------------------- #
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET]

    # --- Point MLflow at the local sqlite tracking store ----------------- #
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"flaml_automl_{int(time.time())}") as run:
        mlflow.log_params({
            "time_budget_sec": time_budget,
            "seed": config.SEED,
            "n_features": len(config.FEATURES),
            "n_train": len(X_train),
            "framework": "FLAML",
            "task": "regression",
        })

        # --- Fit preprocessor, then let FLAML pick & tune the algorithm --- #
        pre = build_preprocessor()
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)

        automl = AutoML()
        automl.fit(
            X_train=X_train_t, y_train=y_train.values,
            task="regression",
            metric="rmse",
            time_budget=time_budget,
            estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
            seed=config.SEED,
            verbose=1,
        )

        # --- AutoML results -------------------------------------------------- #
        best_algo = automl.best_estimator
        best_config = automl.best_config
        print(f"\nAutoML chose: {best_algo}")
        print(f"Best hyperparameters: {json.dumps(best_config, default=str)}")
        mlflow.log_param("automl_best_algorithm", best_algo)
        mlflow.log_dict(
            {k: str(v) for k, v in best_config.items()}, "best_hyperparameters.json"
        )

        # --- Assemble the deployable pipeline (raw input -> prediction) ---- #
        # pre is already fitted; automl.model.estimator is the fitted best model.
        final_pipeline = Pipeline([("preprocessor", pre),
                                   ("model", automl.model.estimator)])

        # --- Evaluate on the held-out test set ----------------------------- #
        preds = final_pipeline.predict(X_test)
        metrics = {
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_mae": float(mean_absolute_error(y_test, preds)),
            "test_r2": float(r2_score(y_test, preds)),
        }
        mlflow.log_metrics(metrics)
        print(f"Test metrics: {json.dumps(metrics, indent=2)}")

        # --- Log + register the model in MLflow ---------------------------- #
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            registered_model_name=config.REGISTERED_MODEL_NAME,
            input_example=X_train.head(2),
        )

        # --- Also save a plain pickle for the Streamlit/FastAPI app -------- #
        # (the app should not depend on the MLflow server being up at demo time)
        joblib.dump(final_pipeline, config.MODEL_PATH)
        meta = {
            "registered_model": config.REGISTERED_MODEL_NAME,
            "best_algorithm": best_algo,
            "features": config.FEATURES,
            "target": config.TARGET,
            "risk_threshold": config.RISK_THRESHOLD,
            "test_metrics": metrics,
            "mlflow_run_id": run.info.run_id,
        }
        config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))

        print(f"\nSaved model -> {config.MODEL_PATH.relative_to(config.ROOT)}")
        print(f"Saved metadata -> {config.MODEL_META_PATH.relative_to(config.ROOT)}")
        print(f"MLflow run id: {run.info.run_id}")
        return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=int, default=60,
                    help="AutoML search time in seconds (60 is fine for the demo)")
    args = ap.parse_args()
    main(time_budget=args.time_budget)
