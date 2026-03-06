# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-spatial: Run Local Tests (no model fitting)

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/burningcost/insurance-spatial.git" polars scipy numpy xarray pytest

# COMMAND ----------

import subprocess
import sys
import os

# Clone the repo to get the test files
clone = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-spatial.git", "/tmp/ins-spatial"],
    capture_output=True, text=True
)
print("Clone:", clone.returncode, clone.stderr[:200] if clone.returncode != 0 else "OK")

# COMMAND ----------

env = {**os.environ, "PYTHONPATH": "/tmp/ins-spatial/src"}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/tmp/ins-spatial/tests/test_adjacency.py",
        "/tmp/ins-spatial/tests/test_diagnostics.py",
        "/tmp/ins-spatial/tests/test_relativities.py",
        "-v",
        "--tb=long",
        "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/tmp/ins-spatial",
    env=env,
)

full_output = result.stdout + "\n--- STDERR ---\n" + result.stderr
# Truncate for display but keep the most useful part (tail)
display_output = full_output[-8000:] if len(full_output) > 8000 else full_output
print(display_output)

# COMMAND ----------

# Exit with the output so it appears in notebook_output.result
exit_message = f"returncode={result.returncode}\n" + display_output[-3000:]
dbutils.notebook.exit(exit_message)
