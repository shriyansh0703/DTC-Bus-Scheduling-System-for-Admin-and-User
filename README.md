# DTC SIH — Bus Data & Route Tools

Small collection of scripts for analyzing Delhi Transport Corporation (DTC) bus data, route-finding, and revenue estimation.

## Project structure
- Python scripts: admin_control.py, user_control.py, api.py, find_shortest_route.py, revenue_generated.py, database.py
- Data: CSV files in project root (e.g., bus_routes_data.csv, bus_revenue_data.csv, bus_passengers_delhi.csv)

## Setup
This project includes a local virtual environment in `env/`. On Windows PowerShell, activate it with:

```powershell
& "env\\Scripts\\Activate.ps1"
```

If you prefer creating a new venv instead:

```powershell
python -m venv env
& "env\\Scripts\\Activate.ps1"
pip install -r requirements.txt  # if you create one
```

Note: The provided `env/` already contains commonly used packages (pandas, numpy, networkx, etc.).

## Usage
- Run an admin script:

```powershell
python admin_control.py
```

- Run the user interface script:

```powershell
python user_control.py
```

- Run the API server (if implemented in `api.py`):

```powershell
python api.py
```

- Compute routes or revenue examples:

```powershell
python find_shortest_route.py
python revenue_generated.py
```

## Data files
Keep CSV files in the project root. Example files included:
- bus_routes_data.csv
- bus_revenue_data.csv
- bus_passengers_delhi.csv
- bus_passengers_estimated_fare_time.csv

