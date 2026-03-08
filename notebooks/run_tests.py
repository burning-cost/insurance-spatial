# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-spatial: Run Tests on Databricks
# MAGIC
# MAGIC This notebook installs the library from GitHub and runs the full pytest suite.
# MAGIC The integration tests (BYM2 model fitting) are enabled here since PyMC is available.

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/burning-cost/insurance-spatial.git" pymc arviz polars xarray pytest

# COMMAND ----------

import subprocess
import sys

# Clone the repo to get the test files
result = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burning-cost/insurance-spatial.git", "/tmp/insurance-spatial"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

# Run the full test suite.
# Integration tests (BYM2 fitting) are enabled because PyMC is installed here.
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/tmp/insurance-spatial/tests/",
        "-v",
        "--tb=short",
        # Disable the integration test skip flag
        "-p", "no:skip",
    ],
    capture_output=True, text=True,
    cwd="/tmp/insurance-spatial",
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"Tests failed with return code {result.returncode}")
else:
    print("\nAll tests passed.")
