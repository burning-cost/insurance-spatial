"""
Run pytest for insurance-spatial on Databricks serverless compute.

Usage:
    python run_databricks_pytest.py
"""

import os
import sys
import time
import base64
import pathlib

# Load Databricks credentials
env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/Shared/insurance-spatial-test"
REPO_ROOT = pathlib.Path(__file__).parent

# ── 1. Upload source files ────────────────────────────────────────────────────

def upload_file(local_path: pathlib.Path, workspace_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(workspace_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=workspace_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )


def upload_tree(local_dir: pathlib.Path, workspace_base: str, extensions: set) -> int:
    count = 0
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and f.suffix in extensions and "__pycache__" not in str(f):
            rel = f.relative_to(local_dir.parent)
            ws_path = f"{workspace_base}/{rel}"
            upload_file(f, ws_path)
            count += 1
    return count


print("Uploading source and tests...")
n = upload_tree(REPO_ROOT / "src", WORKSPACE_DIR, {".py"})
n += upload_tree(REPO_ROOT / "tests", WORKSPACE_DIR, {".py"})
upload_file(REPO_ROOT / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
print(f"  Uploaded {n} files")

# ── 2. Build notebook content ─────────────────────────────────────────────────

NOTEBOOK_CONTENT = f"""\
# Databricks notebook source
# MAGIC %pip install scipy polars numpy pandas pyarrow pytest scikit-learn

# COMMAND ----------

import subprocess, sys, os, shutil, tempfile

WORKSPACE_SRC = "{WORKSPACE_DIR}"

# Use a temp directory to avoid permission issues with workspace __pycache__
LOCAL_DIR = tempfile.mkdtemp(prefix="ins_spatial_")
print(f"Working directory: {{LOCAL_DIR}}")

def copy_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for fname in files:
            if not fname.endswith('.pyc'):
                shutil.copy2(os.path.join(root, fname), os.path.join(dest_dir, fname))

copy_dir(WORKSPACE_SRC + "/src", LOCAL_DIR + "/src")
copy_dir(WORKSPACE_SRC + "/tests", LOCAL_DIR + "/tests")
shutil.copy2(WORKSPACE_SRC + "/pyproject.toml", LOCAL_DIR + "/pyproject.toml")
print(f"Copied project to {{LOCAL_DIR}}")

# Run pytest - skip tests that need optional heavy deps (pymc, arviz)
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     LOCAL_DIR + "/tests",
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
     "--ignore=" + LOCAL_DIR + "/tests/test_model_integration.py",
     f"--rootdir={{LOCAL_DIR}}"],
    capture_output=True, text=True,
    cwd=LOCAL_DIR,
    env={{**os.environ, "PYTHONPATH": LOCAL_DIR + "/src"}}
)

output = result.stdout
if result.stderr:
    output += "\\nSTDERR:\\n" + result.stderr[:500]
output += f"\\nExit code: {{result.returncode}}"

print(output[-10000:])  # print for logs

# Clean up
shutil.rmtree(LOCAL_DIR, ignore_errors=True)

status = "PASSED" if result.returncode == 0 else f"FAILED (exit {{result.returncode}})"
dbutils.notebook.exit(f"{{status}}\\n{{output[-8000:]}}")
"""

NOTEBOOK_PATH = f"{WORKSPACE_DIR}/_run_tests"
content_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
try:
    w.workspace.mkdirs(path=WORKSPACE_DIR)
except Exception:
    pass
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=content_b64,
    overwrite=True,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
)
print(f"Uploaded test notebook to {NOTEBOOK_PATH}")

# ── 3. Submit job using serverless compute ────────────────────────────────────

run_response = w.jobs.submit(
    run_name="insurance-spatial-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=jobs.compute.Environment(
                client="2",
                dependencies=[
                    "scipy",
                    "polars",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "pytest",
                    "scikit-learn",
                    "xarray",
                    "arviz",
                ],
            ),
        )
    ],
)
run_id = run_response.run_id
print(f"\nSubmitted serverless run ID: {run_id}")

# ── 4. Poll until done ────────────────────────────────────────────────────────

while True:
    state = w.jobs.get_run(run_id=run_id)
    life = str(state.state.life_cycle_state)
    result_state = str(state.state.result_state)
    print(f"  Status: {life} / {result_state}      ", end="\r")
    if any(x in life for x in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR")):
        print()
        break
    time.sleep(15)

# ── 5. Fetch output ───────────────────────────────────────────────────────────

for task in state.tasks or []:
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        print("\n" + "=" * 60)
        print("TEST OUTPUT:")
        print("=" * 60)
        if output.notebook_output and output.notebook_output.result:
            print(output.notebook_output.result)
        elif output.error:
            print("ERROR:", output.error)
            if output.error_trace:
                print("TRACE:", output.error_trace[:3000])
        else:
            print("(no output captured)")
    except Exception as e:
        print(f"Could not fetch output for task {task.task_key}: {e}")

final = str(state.state.result_state)
print(f"\nFinal result: {final}")
sys.exit(0 if "SUCCESS" in final else 1)
