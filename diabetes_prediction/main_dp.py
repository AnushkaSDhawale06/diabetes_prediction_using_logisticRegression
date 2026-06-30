"""Main entry point for the diabetes prediction project."""

from pathlib import Path
import subprocess
import sys


def run_script(script_path: Path) -> None:
    """Run one Python script and stop if it fails."""
    print(f"\n[Step] Running {script_path.name}...")
    subprocess.run(
        [sys.executable, str(script_path.name)], check=True, cwd=script_path.parent
    )


if __name__ == "__main__":
    print("Diabetes Prediction Project")
    print("-" * 32)
    print("This launcher runs the full workflow in the right order.")

    # Find the folder that contains this file so the paths work from anywhere.
    project_root = Path(__file__).resolve().parent
    diabetes_project_dir = project_root / "diabetes_project"

    # Step 1: create the custom dataset before any model training begins.
    run_script(diabetes_project_dir / "generate_dataset.py")

    # Step 2: train the model, create charts, and print predictions.
    run_script(diabetes_project_dir / "diabetes_model.py")

    print("\n[Done] The diabetes prediction workflow finished successfully.")
    print(
        "You can now review the CSV file, charts, and model output in the project folder."
    )
