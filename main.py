from nbconvert import PythonExporter
import os

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        # Use nbconvert to convert the notebook to Python code
        exporter = PythonExporter()
        python_code, _ = exporter.from_file(f)

    # Execute the Python code in the current namespace
    exec(python_code, globals())

if __name__ == "__main__":
    # Run different notebooks by passing the paths
    notebook1_path = "GoogleMapAPI.ipynb"
    run_notebook(notebook1_path)

