"""
Launcher: installs npm packages if needed, starts the React UI, then starts the Flask server.
Run from anywhere: python src/run_ui.py
"""
import atexit
import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
UI   = os.path.join(ROOT, "catanatron", "ui")

# server.py uses relative paths (./models/saved/) so cwd must be src/
os.chdir(SRC)
sys.path.insert(0, SRC)

if not os.path.exists(os.path.join(UI, "node_modules")):
    print("[run_ui] Running npm install...")
    subprocess.run("npm install", cwd=UI, shell=True, check=True)

print("[run_ui] Starting React UI at http://localhost:3000")
ui_proc = subprocess.Popen("npm start", cwd=UI, shell=True)

atexit.register(ui_proc.terminate)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

print("[run_ui] Starting Flask server at http://localhost:5001")
from server import app
app.run(host="0.0.0.0", port=5001, debug=False)
