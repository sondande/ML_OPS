"""
monitoring_evidently.py  --  Evidently drift + data quality report
==============================================================================
Uses DataDriftPreset + DataSummaryPreset for Evidently >=0.7.0.

Migration note (0.6.x -> 0.7.x):
  - Report.run() no longer accepts raw pandas DataFrames.
  - Data must be wrapped in evidently.Dataset with a DataDefinition.
  - Evidently Cloud v2 requires evidently>=0.7; CloudWorkspace API updated.

Cloud upload is token-gated: set EVIDENTLY_API_TOKEN env var to enable.
Offline / no-token path always works and writes a local HTML report.

Run:
    python -m src.monitoring_evidently
"""

from pathlib import Path

import pandas as pd

from src import config


def _build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """
    Build and run an Evidently report using the Evidently >=0.7.0 API.

    Key change from 0.6.x: Report.run() now requires evidently.Dataset
    objects (wrapping pandas DataFrames with a DataDefinition) rather
    than accepting raw DataFrames directly.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    # DataDefinition() with no arguments triggers auto-detection of column
    # types (numeric vs categorical) from the DataFrame dtypes.
    data_definition = DataDefinition()

    ref_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset(),
        ]
    )

    return report.run(reference_data=ref_dataset, current_data=cur_dataset)


def _save_local(snapshot, out: Path) -> Path:
    """
    Save Evidently report to a local HTML file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    return out


def _upload_cloud(snapshot) -> bool:
    """
    Try to upload the Evidently report to Evidently Cloud v2.

    Evidently Cloud v2 requires evidently>=0.7.0. The workspace API
    changed between v1 and v2; this function targets the v2 interface.
    If anything fails, continue in local-only mode.
    """
    token = getattr(config, "EVIDENTLY_API_TOKEN", None)
    project_id = getattr(config, "EVIDENTLY_PROJECT_ID", None)
    url = getattr(config, "EVIDENTLY_CLOUD_URL", None)

    if not token:
        return False

    try:
        from evidently.ui.workspace.cloud import CloudWorkspace  # evidently>=0.7
    except ImportError:
        try:
            from evidently.ui.workspace import CloudWorkspace  # evidently 0.6 fallback
        except ImportError:
            print("Evidently Cloud: CloudWorkspace not available in this version.")
            return False

    try:
        ws = CloudWorkspace(token=token, url=url) if url else CloudWorkspace(token=token)

        if project_id:
            ws.add_run(project_id, snapshot)
            print(f"Evidently Cloud: uploaded to project {project_id}")
            return True

        projects = ws.list_projects()

        if not projects:
            print("Evidently Cloud: no project found; set EVIDENTLY_PROJECT_ID")
            return False

        ws.add_run(projects[0].id, snapshot)
        print(f"Evidently Cloud: uploaded to project {projects[0].name}")
        return True

    except Exception as exc:
        print(f"Evidently Cloud upload failed; continuing offline: {exc}")
        return False


def _load_feature_frame(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    """
    Load only model feature columns from a CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")

    df = pd.read_csv(path)

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required feature columns: {missing}")

    return df[feature_cols].copy()


def run() -> Path:
    """
    Generate local Evidently monitoring report and optionally upload to cloud.
    """
    config.ensure_dirs()

    reference = _load_feature_frame(config.TEST_PATH, config.FEATURES)
    current = _load_feature_frame(config.TEST_MODIFIED_PATH, config.FEATURES)

    snapshot = _build_report(reference, current)

    out = config.MONITORING_DIR / "evidently_report.html"
    _save_local(snapshot, out)

    try:
        display_path = out.relative_to(config.ROOT)
    except ValueError:
        display_path = out

    print(f"Evidently report local -> {display_path}")

    if getattr(config, "EVIDENTLY_API_TOKEN", None):
        uploaded = _upload_cloud(snapshot)
        if not uploaded:
            print("Cloud upload not completed; local report is the primary output.")
    else:
        print("EVIDENTLY_API_TOKEN not set — local mode only.")

    return out


if __name__ == "__main__":
    run()
