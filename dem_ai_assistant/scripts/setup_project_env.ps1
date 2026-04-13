# Creates .venv in the project root and installs Python dependencies from requirements.txt.
# Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts/setup_project_env.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at .venv ..."
    python -m venv .venv
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
}

Write-Host "Upgrading pip and installing requirements ..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host ""
Write-Host "Done. Activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Project Chrono (PyChrono) is NOT installed via pip (PyPI has the wrong name)."
Write-Host "For the real DEM/multibody solver, use a separate conda env, for example:"
Write-Host "  conda create -n chrono python=3.11 -y"
Write-Host "  conda activate chrono"
Write-Host "  conda install projectchrono::pychrono -c conda-forge -y"
Write-Host "  conda install numpy pandas pyyaml matplotlib scipy pydantic pytest -c conda-forge -y"
