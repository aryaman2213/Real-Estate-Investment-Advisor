"""Open the MLflow UI on this project's experiments and model registry.

    python mlflow_ui.py            # then open http://127.0.0.1:5000

MLflow stores absolute artifact paths in mlflow.db, so a fresh clone in another folder would show runs and metrics
but not the logged models and plots. This script first rewrites those paths to this folder, then starts the UI.
"""
import sqlite3
import subprocess
import sys

import config

# (table, column) pairs where MLflow keeps absolute artifact paths.
PATH_COLUMNS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
]


def relocate_store(db_path=config.ROOT / "mlflow.db") -> str | None:
    """Point every stored artifact path at this checkout. Returns the old root if anything changed."""
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT artifact_location FROM experiments WHERE artifact_location LIKE '%/mlruns/%' AND name != 'Default' LIMIT 1"
    ).fetchone()
    if row is None:
        con.close()
        return None
    old_root = row[0].removeprefix("file://").split("/mlruns/")[0]
    new_root = str(config.ROOT)
    if old_root != new_root:
        with con:
            for table, column in PATH_COLUMNS:
                con.execute(f"UPDATE {table} SET {column} = replace({column}, ?, ?)", (old_root, new_root))
    con.close()
    return old_root if old_root != new_root else None


def main():
    moved_from = relocate_store()
    if moved_from:
        print(f"Updated MLflow artifact paths: {moved_from} -> {config.ROOT}")
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", config.MLFLOW_TRACKING_URI,
                    "--port", port], cwd=config.ROOT, check=False)


if __name__ == "__main__":
    main()
