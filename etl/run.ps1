# etl\run.ps1

# 1) Figure out where this script lives
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# 2) Path to the venv’s python executable
$VenvPython = Join-Path $ScriptDir '..\kplerenv\Scripts\python.exe'

# 3) Run the fetch script with that Python
& $VenvPython (Join-Path $ScriptDir 'kpler_fetch.py')
