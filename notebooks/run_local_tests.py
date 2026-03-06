# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-spatial: Run Local Tests (no model fitting)

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/burningcost/insurance-spatial.git" polars scipy numpy xarray pytest

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-spatial.git", "/tmp/insurance-spatial-local"],
    capture_output=True, text=True
)
print("Clone stdout:", result.stdout)
print("Clone stderr:", result.stderr)

# COMMAND ----------

# Run only the fast tests (skip integration tests)
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/tmp/insurance-spatial-local/tests/test_adjacency.py",
        "/tmp/insurance-spatial-local/tests/test_diagnostics.py",
        "/tmp/insurance-spatial-local/tests/test_relativities.py",
        "-v",
        "--tb=long",
    ],
    capture_output=True, text=True,
    cwd="/tmp/insurance-spatial-local",
    env={**__import__('os').environ, "PYTHONPATH": "/tmp/insurance-spatial-local/src"},
)

# Always print full output regardless of exit code
output = result.stdout + "\n" + result.stderr
print(output[-10000:] if len(output) > 10000 else output)

if result.returncode != 0:
    raise RuntimeError(f"Tests failed with return code {result.returncode}")
else:
    print("\nAll fast tests passed.")
