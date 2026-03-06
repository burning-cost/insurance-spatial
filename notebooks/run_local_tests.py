# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-spatial: Run Local Tests (no model fitting)
# MAGIC
# MAGIC Runs only the fast tests that don't require MCMC (adjacency, diagnostics, relativities).
# MAGIC These test the pure-Python/numpy/scipy code paths.

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/burningcost/insurance-spatial.git" polars scipy numpy xarray pytest

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-spatial.git", "/tmp/insurance-spatial-local"],
    capture_output=True, text=True
)
print(result.stdout or "Cloned.")
print(result.stderr)

# COMMAND ----------

# Run only the fast tests (skip integration tests)
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/tmp/insurance-spatial-local/tests/test_adjacency.py",
        "/tmp/insurance-spatial-local/tests/test_diagnostics.py",
        "/tmp/insurance-spatial-local/tests/test_relativities.py",
        "-v",
        "--tb=short",
    ],
    capture_output=True, text=True,
    cwd="/tmp/insurance-spatial-local",
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"Tests failed with return code {result.returncode}")
else:
    print("\nAll fast tests passed.")
