# tested for 3.8
PYTHON_EXEC="python3.8"
ELLOGON_PYTHON_PKG_PATH="/home/sthemeli/projects/ellogon"

$PYTHON_EXEC -m venv venv
venv/bin/python -m pip install -r requirements.txt
$PYTHON_EXEC -m pip install "$ELLOGON_PYTHON_PKG_PATH"
