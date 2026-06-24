## Update catanatron:
```bash
git submodule update --init --recursive
```

## Create the python venv:
```bash
python -m venv venv
```

## Activate teh python venv:
**Windows:**
```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\venv\Scripts\Activate.ps1)
```
**Linux:**
```bash
. ./venv/bin/activate
```

## Install the requirements:
```bash
cd catanatron && pip install -e .[web,gym,dev] && cd ..
pip install -r ./requirements.txt
```

## Run the tests:
```bash
cd src && python -m pytest tests/ && cd ..
```

## UI
Requires [Node.js v24](https://nodejs.org).
```bash
python src/ui/run_ui.py
```


## Dashboard
```bash
cd src && streamlit run dashboard.py && cd ..
```